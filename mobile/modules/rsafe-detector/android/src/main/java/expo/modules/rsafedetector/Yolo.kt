// Runs an Ultralytics YOLO .onnx export on a Bitmap (letterbox -> CHW float -> boxes).
package expo.modules.rsafedetector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.roundToInt

class Yolo(
  private val env: OrtEnvironment,
  modelBytes: ByteArray,
  private val numClasses: Int,
  threads: Int,
  // The 8-bit models were measured on the phone with ONNX Runtime's default CPU kernels
  // (vehicle 87 ms, helmet 42 ms); XNNPACK is kept as an option, not assumed faster.
  useXnnpack: Boolean = false,
) : AutoCloseable {
  private val session: OrtSession
  private val inputName: String
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  // Reused per input size
  private var size = 0
  private lateinit var canvasBitmap: Bitmap
  private lateinit var pixels: IntArray
  private lateinit var input: FloatBuffer

  init {
    val opts = OrtSession.SessionOptions()
    opts.setIntraOpNumThreads(threads)
    opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    if (useXnnpack) {
      try {
        opts.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
      } catch (e: Exception) {
        Log.w("RSAFE", "XNNPACK not available, using default CPU", e)
      }
    }
    session = env.createSession(modelBytes, opts)
    inputName = session.inputNames.first()
  }

  private fun prepare(s: Int) {
    if (s == size) return
    size = s
    canvasBitmap = Bitmap.createBitmap(s, s, Bitmap.Config.ARGB_8888)
    pixels = IntArray(s * s)
    input = ByteBuffer.allocateDirect(3 * s * s * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
  }

  /**
   * Detect in `image` (or in `region` of it). Boxes come back in `image` coordinates.
   * keep = class ids to keep (null = all). Class-aware NMS like Ultralytics.
   */
  @Synchronized
  fun detect(
    image: Bitmap,
    inputSize: Int,
    conf: Float,
    iou: Float,
    keep: Set<Int>? = null,
    region: Rect? = null,
  ): List<Det> {
    prepare(inputSize)
    val src = region ?: Rect(0, 0, image.width, image.height)
    if (src.width() <= 1 || src.height() <= 1) return emptyList()

    // Letterbox: keep aspect ratio, pad with grey 114 (same as the laptop's HelmetDetector)
    val scale = min(inputSize.toFloat() / src.width(), inputSize.toFloat() / src.height())
    val nw = (src.width() * scale).roundToInt()
    val nh = (src.height() * scale).roundToInt()
    val padX = (inputSize - nw) / 2
    val padY = (inputSize - nh) / 2
    val canvas = Canvas(canvasBitmap)
    canvas.drawColor(Color.rgb(114, 114, 114))
    canvas.drawBitmap(image, src, Rect(padX, padY, padX + nw, padY + nh), paint)
    canvasBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

    val plane = inputSize * inputSize
    input.rewind()
    for (i in 0 until plane) {
      val p = pixels[i]
      input.put(i, ((p shr 16) and 0xFF) / 255f)
      input.put(plane + i, ((p shr 8) and 0xFF) / 255f)
      input.put(2 * plane + i, (p and 0xFF) / 255f)
    }

    val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
    val out = ArrayList<Det>()
    OnnxTensor.createTensor(env, input, shape).use { tensor ->
      session.run(mapOf(inputName to tensor)).use { result ->
        val output = result.get(0) as OnnxTensor
        val dims = output.info.shape // [1, 4 + nc, N]
        val features = dims[1].toInt()
        val n = dims[2].toInt()
        require(features == 4 + numClasses) { "Unexpected model output $features, expected ${4 + numClasses}" }
        val buf = output.floatBuffer
        for (a in 0 until n) {
          var bestCls = -1
          var best = conf
          for (c in 0 until numClasses) {
            if (keep != null && c !in keep) continue
            val s = buf.get((4 + c) * n + a)
            if (s >= best) {
              best = s
              bestCls = c
            }
          }
          if (bestCls < 0) continue
          val cx = buf.get(a)
          val cy = buf.get(n + a)
          val w = buf.get(2 * n + a)
          val h = buf.get(3 * n + a)
          val x1 = ((cx - w / 2 - padX) / scale + src.left).coerceIn(src.left.toFloat(), src.right.toFloat())
          val y1 = ((cy - h / 2 - padY) / scale + src.top).coerceIn(src.top.toFloat(), src.bottom.toFloat())
          val x2 = ((cx + w / 2 - padX) / scale + src.left).coerceIn(src.left.toFloat(), src.right.toFloat())
          val y2 = ((cy + h / 2 - padY) / scale + src.top).coerceIn(src.top.toFloat(), src.bottom.toFloat())
          if (x2 <= x1 || y2 <= y1) continue
          out.add(Det(Box(x1, y1, x2, y2), best, bestCls))
        }
      }
    }
    return nms(out, iou)
  }

  override fun close() = session.close()
}
