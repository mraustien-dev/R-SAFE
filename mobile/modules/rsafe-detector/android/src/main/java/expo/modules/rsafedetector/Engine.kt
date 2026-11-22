// Models, plate reading and the evidence writer. The per-frame flow is in Pipeline.kt.
package expo.modules.rsafedetector

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class EngineSettings(
  // Input sizes must match the 8-bit model files (ai-engine/tools/export_mobile_models.py)
  val vehicleSize: Int = 480,
  val detectConf: Float = 0.10f,  // weak motorcycles are kept: they may continue a track
  val personConf: Float = 0.35f,
  val helmetSize: Int = 320,
  val helmetConf: Float = 0.25f,
  val plateSize: Int = 640,
  val plateConf: Float = 0.25f,
  val minOcrConfidence: Float = 0.85f, // ML Kit scale; tune on real plates
  val expandRatio: Float = 0.15f,
  val maxRiders: Int = 3,
  val headRatio: Float = 0.40f,
  val padRatio: Float = 0.20f,
  val minRiderHeightPx: Float = 48f,
  val threadsPerModel: Int = 2, // tracker and helmet worker run at the same time
  val rules: RulesConfig = RulesConfig(),
  val tracker: TrackerConfig = TrackerConfig(),
)

/** Evidence of the best moment: the full frame (never modified), the rider crop, all rider boxes. */
class Frames(val frame: Bitmap, val rider: Bitmap, val riderBoxes: List<Box>)

class Engine(context: Context, val settings: EngineSettings = EngineSettings()) : AutoCloseable {
  companion object {
    const val PERSON = 0
    const val MOTORCYCLE = 3
  }

  private val env = OrtEnvironment.getEnvironment()
  private val vehicle = Yolo(env, asset(context, "models/vehicle.onnx"), 80, settings.threadsPerModel)
  private val helmet = Yolo(env, asset(context, "models/helmet.onnx"), 2, settings.threadsPerModel)
  private val plate = Yolo(env, asset(context, "models/plate.onnx"), 1, settings.threadsPerModel)
  private val ocr = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
  val evidenceRoot = File(context.filesDir, "evidence").apply { mkdirs() }

  /** Which model files made a violation (from models.json: file hashes), saved in every record. */
  val models: Any = try {
    val m = JSONObject(String(asset(context, "models/models.json"))).getJSONObject("models")
    val out = JSONObject()
    for (k in m.keys()) {
      val e = m.getJSONObject(k)
      out.put(k, JSONObject().put("file", e.optString("file")).apply {
        if (e.has("sha256")) put("sha256", e.getString("sha256"))
        if (e.has("input_size")) put("input_size", e.getInt("input_size"))
      })
    }
    out
  } catch (e: Exception) {
    Log.w("RSAFE", "models.json missing", e)
    JSONObject().put("vehicle", "vehicle.onnx").put("helmet", "helmet.onnx").put("plate", "plate.onnx")
  }

  private fun asset(context: Context, path: String) = context.assets.open(path).use { it.readBytes() }

  /** Motorcycles (all, weak ones included) and confident persons. */
  fun detectVehicles(frame: Bitmap): Pair<List<Det>, List<Det>> {
    val dets = vehicle.detect(frame, settings.vehicleSize, settings.detectConf, 0.7f, setOf(PERSON, MOTORCYCLE))
    return dets.filter { it.cls == MOTORCYCLE } to dets.filter { it.cls == PERSON && it.conf >= settings.personConf }
  }

  fun classifyHelmet(frame: Bitmap, head: Rect): HelmetResult? {
    val dets = helmet.detect(frame, settings.helmetSize, settings.helmetConf, 0.45f, region = head)
    val best = dets.maxByOrNull { it.conf } ?: return null
    return HelmetResult(if (best.cls == 0) "helmet" else "no_helmet", best.conf)
  }

  /** Plate in a motorcycle region crop (plateSearchRegion): detect, then OCR. */
  fun readPlate(region: Bitmap): PlateResult<Bitmap> {
    val best = plate.detect(region, settings.plateSize, settings.plateConf, 0.7f)
      .maxByOrNull { it.conf } ?: return PlateResult("not_visible")
    val crop = crop(region, best.box)
    if (crop.width < 2 || crop.height < 2) return PlateResult("not_visible")
    return try {
      val (raw, conf) = ocrText(upscale(crop))
      val text = normalizePlateText(raw)
      if (text.isEmpty() || conf < settings.minOcrConfidence) {
        PlateResult("unreadable", crop = crop)
      } else {
        PlateResult("read", text, (conf * 100).roundToInt() / 100f, crop)
      }
    } catch (e: Exception) {
      Log.w("RSAFE", "OCR failed", e)
      PlateResult("unreadable", crop = crop)
    }
  }

  /** Lines top to bottom, each left to right; confidence = the weakest line. */
  private fun ocrText(bitmap: Bitmap): Pair<String, Float> {
    val result = Tasks.await(ocr.process(InputImage.fromBitmap(bitmap, 0)))
    val lines = result.textBlocks.flatMap { it.lines }
      .sortedWith(compareBy({ it.boundingBox?.centerY() ?: 0 }, { it.boundingBox?.left ?: 0 }))
    if (lines.isEmpty()) return "" to 0f
    val text = lines.joinToString("") { it.text }
    val conf = lines.minOf { it.confidence }
    return text to conf
  }

  private fun upscale(crop: Bitmap): Bitmap {
    if (crop.height >= 96) return crop
    val s = min(96f / crop.height, 4f)
    return Bitmap.createScaledBitmap(crop, (crop.width * s).roundToInt(), (crop.height * s).roundToInt(), true)
  }

  fun crop(frame: Bitmap, box: Box): Bitmap {
    val l = max(0, box.x1.toInt()).coerceAtMost(frame.width - 1)
    val t = max(0, box.y1.toInt()).coerceAtMost(frame.height - 1)
    val r = min(frame.width, box.x2.roundToInt())
    val b = min(frame.height, box.y2.roundToInt())
    return Bitmap.createBitmap(frame, l, t, max(1, r - l), max(1, b - t))
  }

  // -------------------------------------------------------------------------------------
  // Evidence: frame.jpg (boxes drawn), rider.jpg, plate.jpg in app-private storage.
  // The SHA-256 of each file goes into the record the phone signs, so any later change
  // to a photo is detected by the server.

  fun save(report: TrackReport<Frames, Bitmap>, plate: PlateResult<Bitmap>): String? = try {
    val dir = File(evidenceRoot, UUID.randomUUID().toString()).apply { mkdirs() }
    val annotated = report.evidence.frame.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(annotated)
    val stroke = max(2f, annotated.width / 320f)
    val blue = Paint().apply { style = Paint.Style.STROKE; strokeWidth = stroke; color = Color.rgb(0, 160, 255) }
    val red = Paint().apply { style = Paint.Style.STROKE; strokeWidth = stroke; color = Color.RED }
    report.motorcycle.box.let { canvas.drawRect(it.x1, it.y1, it.x2, it.y2, blue) }
    report.evidence.riderBoxes.forEach { canvas.drawRect(it.x1, it.y1, it.x2, it.y2, red) }

    val images = LinkedHashMap<String, Any?>()
    images["frame"] = writeJpeg(annotated, File(dir, "frame.jpg"))
    images["rider"] = writeJpeg(report.evidence.rider, File(dir, "rider.jpg"))
    plate.crop?.let { images["plate"] = writeJpeg(it, File(dir, "plate.jpg")) }

    val event = mapOf(
      "trackId" to report.trackId,
      "detectedAtMs" to report.seenAtMs.toDouble(),
      "folder" to dir.absolutePath,
      "riders" to report.riders.map {
        mapOf(
          "position" to it.position,
          "helmet" to it.helmet,
          "confidence" to it.confidence.toDouble(),
          "frames_observed" to it.framesObserved,
        )
      },
      "motorcycle" to mapOf(
        "confidence" to ((report.motorcycle.conf * 100).roundToInt() / 100.0),
        "bbox" to report.motorcycle.box.toList(),
      ),
      "plate" to mapOf(
        "status" to plate.status,
        "text" to plate.text,
        "ocr_confidence" to plate.conf?.toDouble(),
      ),
      "images" to images,
      "models" to models,
    )
    // Written last: a folder with event.json is complete. The app re-queues any such
    // folder it has no record of (e.g. the app was closed right after detection).
    val json = JSONObject(event).toString()
    File(dir, "event.json").writeText(json)
    json // the app gets exactly the text in event.json
  } catch (e: Exception) {
    Log.e("RSAFE", "Could not save evidence", e)
    null
  }

  /** Size of the last frame.jpg written (bytes), for the on-screen test numbers. */
  @Volatile var lastFrameJpegBytes = 0
    private set

  /**
   * JPEG at quality 90. If that is over the limit (possible for a busy 4K frame), the quality
   * steps down until it fits: the server refuses photos over 5 MB, and a refused photo would
   * block the upload queue.
   */
  private fun writeJpeg(bitmap: Bitmap, file: File): Map<String, Any> {
    var quality = 90
    var bytes: ByteArray
    while (true) {
      bytes = ByteArrayOutputStream().use { out ->
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, out)
        out.toByteArray()
      }
      if (bytes.size <= MAX_JPEG_BYTES || quality <= 50) break
      quality -= 10
    }
    FileOutputStream(file).use { it.write(bytes) }
    if (file.name == "frame.jpg") lastFrameJpegBytes = bytes.size
    return mapOf("path" to file.absolutePath, "sha256" to sha256Hex(bytes), "bytes" to bytes.size)
  }

  override fun close() {
    vehicle.close()
    helmet.close()
    plate.close()
    ocr.close()
  }
}

/** Under the server's 5 MB per-photo limit (device-api MAX_IMAGE_BYTES), with a margin. */
const val MAX_JPEG_BYTES = 4_500_000

fun sha256Hex(bytes: ByteArray): String =
  MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
