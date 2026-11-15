// Native camera view: CameraX preview + two-speed live detection (Pipeline.kt) + box overlay.
package expo.modules.rsafedetector

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.hardware.display.DisplayManager
import android.os.Debug
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.View
import android.widget.FrameLayout
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min
import kotlin.math.roundToInt

/** One engine for the whole app: loading three models takes a few seconds. */
object EngineHolder {
  @Volatile private var engine: Engine? = null

  fun get(context: Context): Engine =
    engine ?: synchronized(this) {
      engine ?: Engine(context.applicationContext).also { engine = it }
    }
}

@SuppressLint("ViewConstructor")
class RsafeCameraView(context: Context, appContext: AppContext) : ExpoView(context, appContext) {
  override val shouldUseAndroidLayout = true

  private val onStatus by EventDispatcher()
  private val onViolation by EventDispatcher()
  private val onError by EventDispatcher()

  private val previewView = PreviewView(context).apply {
    scaleType = PreviewView.ScaleType.FIT_CENTER
    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
  }
  private val overlay = OverlayView(context)
  // [A] tracker loop runs here (CameraX delivers frames on it)
  private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "rsafe-tracker") }

  private var cameraProvider: ProcessCameraProvider? = null
  @Volatile private var active = false
  private var bound = false
  @Volatile private var pipeline: Pipeline? = null
  private var statsStart = SystemClock.elapsedRealtime()
  private var previewUseCase: Preview? = null
  private var analysisUseCase: ImageAnalysis? = null
  private var rotation = Surface.ROTATION_0

  // TEST: "1080p" (default) or "4k" detector frames; the photos come from these frames
  @Volatile private var quality = "1080p"
  // time to turn each camera image into an upright bitmap (grows with resolution)
  private var convertNs = 0L
  private var convertCount = 0
  private var lastLongSide = 0 // analysis thread only

  // Motion-priority exposure (short shutter, sharp fast riders); null = phone's auto exposure
  private var exposure: ExposureController? = null
  private var exposureControl: Camera2CameraControl? = null
  private var lastExposureCheck = 0L

  // Portrait <-> landscape: the activity is not recreated, so tell CameraX the new screen
  // rotation (frames then arrive upright for the detector). Includes 90 -> 270 flips, which
  // do not change the screen size and so do not reach onConfigurationChanged.
  private val displayListener = object : DisplayManager.DisplayListener {
    override fun onDisplayChanged(displayId: Int) = applyRotation()
    override fun onDisplayAdded(displayId: Int) {}
    override fun onDisplayRemoved(displayId: Int) {}
  }

  init {
    val container = FrameLayout(context)
    container.addView(previewView, FrameLayout.LayoutParams(-1, -1))
    container.addView(overlay, FrameLayout.LayoutParams(-1, -1))
    addView(container, LayoutParams(-1, -1))
  }

  override fun onAttachedToWindow() {
    super.onAttachedToWindow()
    (context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager)
      .registerDisplayListener(displayListener, Handler(Looper.getMainLooper()))
  }

  override fun onDetachedFromWindow() {
    (context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager)
      .unregisterDisplayListener(displayListener)
    super.onDetachedFromWindow()
  }

  private fun applyRotation() {
    val r = display?.rotation ?: return
    if (r == rotation) return
    rotation = r
    previewUseCase?.targetRotation = r
    analysisUseCase?.targetRotation = r
    overlay.clear()
    // Box positions change when the frame turns, so a motorcycle tracked across the turn would
    // look like a new one. Hand over what is tracked now (recorded if it already has enough
    // no-helmet votes) and start tracking fresh, instead of risking the same rider twice.
    if (bound) analysisExecutor.execute { pipeline?.flushAll() }
  }

  fun setActive(value: Boolean) {
    if (value == active) return
    active = value
    if (active) start() else stop()
  }

  fun setQuality(value: String) {
    val q = if (value == "4k") "4k" else "1080p"
    if (q == quality) return
    quality = q
    if (!active || !bound) return // used at the next bind
    val owner = appContext.currentActivity as? LifecycleOwner ?: return
    overlay.clear()
    bindCamera(owner) // analyse() starts fresh tracks when the first frame of the new size arrives
  }

  private fun start() {
    val owner = appContext.currentActivity as? LifecycleOwner
    if (owner == null) {
      onError(mapOf("message" to "No activity for the camera"))
      return
    }
    onStatus(mapOf("state" to "loading"))
    // The camera does not wait for the detector: a detector failure must never look like a
    // dead camera. Frames that arrive before the models are ready are simply dropped.
    bindCamera(owner)
    analysisExecutor.execute {
      try {
        val engine = EngineHolder.get(context)
        pipeline = Pipeline(
          engine,
          onViolation = { json -> onViolation(mapOf("json" to json)) },
          onOverlay = { overlay.show(it) },
        )
        statsStart = SystemClock.elapsedRealtime()
        if (active) onStatus(mapOf("state" to "running"))
      } catch (e: Throwable) {
        Log.e("RSAFE", "Engine failed to load", e)
        onStatus(mapOf("state" to "failed"))
        onError(mapOf("message" to "Detector failed to load: ${describe(e)}"))
      }
    }
  }

  /** Full error text for the screen: type + message + root cause, so one screenshot is enough. */
  private fun describe(e: Throwable): String {
    val parts = mutableListOf("${e.javaClass.name}: ${e.message}")
    var cause = e.cause
    while (cause != null && parts.size < 4) {
      parts += "caused by ${cause.javaClass.name}: ${cause.message}"
      cause = if (cause.cause === cause) null else cause.cause
    }
    return parts.joinToString(" / ")
  }

  private fun bindCamera(owner: LifecycleOwner) {
    if (!active) return
    val future = ProcessCameraProvider.getInstance(context)
    future.addListener({
      try {
        val provider = future.get()
        cameraProvider = provider
        rotation = display?.rotation ?: Surface.ROTATION_0
        val preview = Preview.Builder().setTargetRotation(rotation).build()
          .also { it.surfaceProvider = previewView.surfaceProvider }
        // 1080p: the vehicle model sees a 480 px copy (same speed), head and plate crops come
        // from the full frame (2.25x the pixels of 720p for small, distant heads and plates).
        // TEST 4k: 4x the pixels of 1080p for sharper photos; the phone may give less (the
        // screen shows the size it really gave).
        val size = if (quality == "4k") Size(3840, 2160) else Size(1920, 1080)
        val analysis = ImageAnalysis.Builder()
          .setTargetRotation(rotation)
          .setResolutionSelector(
            ResolutionSelector.Builder()
              .setResolutionStrategy(
                ResolutionStrategy(size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER),
              )
              .build(),
          )
          .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
          .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
          .build()
        analysis.setAnalyzer(analysisExecutor) { image -> analyse(image) }
        provider.unbindAll()
        val camera = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        previewUseCase = preview
        analysisUseCase = analysis
        bound = true
        setupExposure(camera)
      } catch (e: Exception) {
        Log.e("RSAFE", "Camera failed", e)
        onError(mapOf("message" to "Camera failed: ${describe(e)}"))
      }
    }, ContextCompat.getMainExecutor(context))
  }

  // ---- motion-priority exposure ---------------------------------------------------------

  @OptIn(ExperimentalCamera2Interop::class)
  private fun setupExposure(camera: Camera) {
    exposure = null
    exposureControl = null
    try {
      val info = Camera2CameraInfo.from(camera.cameraInfo)
      val caps = info.getCameraCharacteristic(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: return
      if (CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR !in caps) return
      val iso = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE) ?: return
      val shutter = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE) ?: return
      val controller = ExposureController(iso.lower..iso.upper, shutter.lower..shutter.upper)
      exposureControl = Camera2CameraControl.from(camera.cameraControl)
      exposure = controller
      applyExposure(controller.current)
    } catch (e: Exception) {
      Log.w("RSAFE", "Manual exposure not available, using auto exposure", e)
      exposure = null
    }
  }

  @OptIn(ExperimentalCamera2Interop::class)
  private fun applyExposure(e: Exposure) {
    exposureControl?.setCaptureRequestOptions(
      CaptureRequestOptions.Builder()
        .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
        .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, e.shutterNs)
        .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, e.iso)
        .setCaptureRequestOption(CaptureRequest.SENSOR_FRAME_DURATION, 33_333_333L)
        .build(),
    )
  }

  /** Twice a second: measure brightness on a tiny copy and nudge shutter / ISO. */
  private fun adjustExposure(frame: Bitmap) {
    val controller = exposure ?: return
    val now = SystemClock.elapsedRealtime()
    if (now - lastExposureCheck < 500) return
    lastExposureCheck = now
    try {
      val small = Bitmap.createScaledBitmap(frame, 64, 36, true)
      val px = IntArray(64 * 36).also { small.getPixels(it, 0, 64, 0, 0, 64, 36) }
      val before = controller.current
      val after = controller.update(meanLuma(px))
      if (after != before) applyExposure(after)
    } catch (e: Exception) {
      Log.w("RSAFE", "Exposure update failed", e)
    }
  }

  private fun exposureText(): String {
    val e = exposure?.current ?: return "auto"
    return "1/${(1_000_000_000.0 / e.shutterNs).roundToInt()} s ISO ${e.iso}"
  }

  // ---- frames ---------------------------------------------------------------------------

  private fun analyse(image: ImageProxy) {
    val pipe = pipeline
    if (!active || pipe == null) {
      image.close()
      return
    }
    try {
      val capturedAt = SystemClock.elapsedRealtime()
      val c0 = System.nanoTime()
      val raw = image.toBitmap()
      val degrees = image.imageInfo.rotationDegrees
      image.close()
      val frame = if (degrees == 0) raw else {
        Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(degrees.toFloat()) }, true)
      }
      convertNs += System.nanoTime() - c0
      convertCount++
      // New resolution (1080p <-> 4K): box sizes change, as after a rotation, so hand over
      // what is tracked and start fresh instead of mixing both sizes in one track.
      val longSide = maxOf(frame.width, frame.height)
      if (lastLongSide != 0 && longSide != lastLongSide) pipe.flushAll()
      lastLongSide = longSide
      adjustExposure(frame)
      pipe.onFrame(frame, System.currentTimeMillis(), capturedAt)
      reportStats(pipe)
    } catch (e: Throwable) {
      Log.e("RSAFE", "Frame skipped", e)
      try { image.close() } catch (_: Exception) {}
    }
  }

  private fun reportStats(pipe: Pipeline) {
    val now = SystemClock.elapsedRealtime()
    val dt = now - statsStart
    if (dt < 2000) return
    val s = pipe.takeStats()
    onStatus(
      mapOf(
        "state" to "running",
        "fps" to s.frames * 1000.0 / dt,
        "vehicleMs" to s.vehicleMs,
        "helmetPerSec" to s.helmetChecks * 1000.0 / dt,
        "helmetMs" to s.helmetMs,
        "helmetLagMs" to s.helmetLagMs,
        "queue" to s.queue,
        "tracks" to s.tracks,
        "plateMs" to s.plateMs,
        "exposure" to exposureText(),
        // TEST numbers for 1080p vs 4K
        "quality" to quality,
        "frameW" to s.frameW,
        "frameH" to s.frameH,
        "convertMs" to if (convertCount > 0) convertNs / 1e6 / convertCount else 0.0,
        // bitmaps live in native memory (Android 8+), so count both heaps
        "memoryMb" to (Debug.getNativeHeapAllocatedSize() +
          Runtime.getRuntime().let { it.totalMemory() - it.freeMemory() }) / 1_048_576.0,
        "photoKb" to s.lastPhotoBytes / 1024.0,
      ),
    )
    convertNs = 0L
    convertCount = 0
    statsStart = now
  }

  private fun stop() {
    cameraProvider?.unbindAll()
    bound = false
    overlay.clear()
    // Motorcycles still in view when duty stops are written now, not lost.
    analysisExecutor.execute {
      pipeline?.let {
        it.flushAll()
        it.close()
      }
      pipeline = null
      onStatus(mapOf("state" to "stopped"))
    }
  }

  fun release() {
    if (active) {
      active = false
      stop()
    }
    analysisExecutor.shutdown()
  }
}

/**
 * Draws motorcycle (blue) and rider (green helmet / red no helmet / grey not checked yet)
 * boxes. Between detections (~10 per second) the boxes keep moving with each motorcycle's
 * measured speed, redrawn every screen frame, so they stay on fast vehicles instead of
 * trailing behind. Stops moving if no new detection arrives for 300 ms.
 */
class OverlayView(context: Context) : View(context) {
  @Volatile private var last: OverlaySnapshot? = null
  private val moto = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 5f; color = Color.rgb(0, 160, 255) }
  private val text = Paint().apply { color = Color.WHITE; textSize = 34f; isFakeBoldText = true }
  private val rider = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 5f }

  fun show(o: OverlaySnapshot) {
    last = o
    postInvalidateOnAnimation()
  }

  fun clear() {
    last = null
    postInvalidate()
  }

  override fun onDraw(canvas: Canvas) {
    val o = last ?: return
    val age = SystemClock.elapsedRealtime() - o.atElapsedMs
    val dt = age.coerceIn(0L, 300L).toFloat()
    // Same mapping as PreviewView FIT_CENTER
    val s = min(width.toFloat() / o.frameW, height.toFloat() / o.frameH)
    val ox = (width - o.frameW * s) / 2f
    val oy = (height - o.frameH * s) / 2f
    fun rect(b: Box, p: Paint) = canvas.drawRect(ox + b.x1 * s, oy + b.y1 * s, ox + b.x2 * s, oy + b.y2 * s, p)
    for (t in o.tracks) {
      val dx = t.vx * dt
      val dy = t.vy * dt
      val box = t.box.shift(dx, dy)
      rect(box, moto)
      canvas.drawText("#${t.id}", ox + box.x1 * s, oy + box.y1 * s - 8f, text)
      for ((r, result) in t.riders) {
        val rb = r.shift(dx, dy)
        rider.color = when (result?.label) {
          "helmet" -> Color.rgb(0, 200, 0)
          "no_helmet" -> Color.RED
          else -> Color.GRAY
        }
        rect(rb, rider)
        result?.let {
          canvas.drawText(
            "${if (it.label == "helmet") "Helmet" else "NO HELMET"} ${"%.2f".format(it.conf)}",
            ox + rb.x1 * s, oy + rb.y2 * s + 34f, text,
          )
        }
      }
    }
    if (age < 300) postInvalidateOnAnimation()
  }
}
