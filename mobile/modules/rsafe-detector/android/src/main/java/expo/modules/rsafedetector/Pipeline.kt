// Two-speed detection pipeline:
//  [A] tracker loop (camera thread): vehicle model -> motion tracker -> riders -> live boxes;
//      offers head crops to [B] and keeps plate candidates. Never waits for [B] or [C].
//  [B] helmet worker (own thread): classifies the freshest head crop per rider, adds votes,
//      keeps the best (largest, sharpest) no-helmet moment as evidence.
//  [C] evidence worker (own thread): for each finished violation reads the plate from the
//      saved candidates, writes + hashes the photos, hands event.json to the app.
package expo.modules.rsafedetector

import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.min

/** What the overlay draws: tracks with their velocity, so boxes can be moved to "now". */
class OverlayTrack(val id: Int, val box: Box, val vx: Float, val vy: Float, val riders: List<Pair<Box, HelmetResult?>>)

class OverlaySnapshot(val frameW: Int, val frameH: Int, val atElapsedMs: Long, val tracks: List<OverlayTrack>)

class Pipeline(
  private val engine: Engine,
  private val onViolation: (String) -> Unit,
  private val onOverlay: (OverlaySnapshot) -> Unit,
) {
  private class HelmetJob(
    val frame: Bitmap,  // full frame, never modified (shared by every job of that frame)
    val head: Rect,
    val rider: Det,
    val riderBoxes: List<Box>,
    val moto: Det,
    val atMs: Long,
  )

  private val s = engine.settings
  private val tracker = MotoTracker(s.tracker)                 // [A] only
  private val rules = ViolationTracker<Frames, Bitmap>(s.rules) // thread-safe
  private val queue = HelmetQueue<HelmetJob>()
  private val evidence: ExecutorService = Executors.newSingleThreadExecutor { r ->
    Thread({ Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND); r.run() }, "rsafe-evidence")
  }
  @Volatile private var running = true

  // ---- stats (read by the camera view every 2 s) ----
  private val frames = AtomicLong()
  private val vehicleNs = AtomicLong()
  private val helmetCount = AtomicLong()
  private val helmetNs = AtomicLong()
  private val helmetLagNs = AtomicLong()
  @Volatile private var lastPlateMs = 0L

  // Started LAST: the thread uses every field above, which must exist before it runs.
  private val helmetThread = Thread({ helmetLoop() }, "rsafe-helmet").apply { start() }

  class Stats(val frames: Long, val vehicleMs: Double, val helmetChecks: Long, val helmetMs: Double,
              val helmetLagMs: Double, val queue: Int, val tracks: Int, val plateMs: Long)

  /** Totals since the last call (then reset). */
  fun takeStats(): Stats {
    val f = frames.getAndSet(0)
    val h = helmetCount.getAndSet(0)
    return Stats(
      f, if (f > 0) vehicleNs.getAndSet(0) / 1e6 / f else 0.0,
      h, if (h > 0) helmetNs.getAndSet(0) / 1e6 / h else 0.0,
      if (h > 0) helmetLagNs.getAndSet(0) / 1e6 / h else 0.0,
      queue.size, rules.openTracks, lastPlateMs,
    )
  }

  // ---------------------------------------------------------------------------------------
  // [A] tracker loop: called on the camera thread for every upright frame.

  /** wallMs: clock time of the frame (records); capturedAtElapsedMs: when it was captured (overlay). */
  fun onFrame(frame: Bitmap, wallMs: Long, capturedAtElapsedMs: Long) {
    val t0 = System.nanoTime()
    val (motos, persons) = engine.detectVehicles(frame)
    vehicleNs.addAndGet(System.nanoTime() - t0)
    frames.incrementAndGet()

    val seen = tracker.update(motos, wallMs)
    val byMoto = associateRiders(seen.map { it.det }, persons, s.expandRatio, s.maxRiders)
    val ridersOf = HashMap<Int, List<Pair<Box, HelmetResult?>>>()

    for ((i, track) in seen.withIndex()) {
      val moto = track.det
      rules.seen(moto, wallMs)
      val riders = byMoto.getValue(i)
      val boxes = riders.map { it.box }
      val last = rules.lastResults(track.id)
      ridersOf[track.id] = riders.mapIndexed { p, r -> r.box to last[p + 1] }

      for ((p, rider) in riders.withIndex()) {
        if (rider.box.height < s.minRiderHeightPx) continue
        val r = headCropRect(rider.box, s.headRatio, s.padRatio, frame.width, frame.height) ?: continue
        queue.offer(RiderKey(track.id, p + 1), HelmetJob(frame, Rect(r[0], r[1], r[2], r[3]), rider, boxes, moto, wallMs))
      }

      // keep the largest views for reading the plate later (only crops, a few per motorcycle)
      val region = plateSearchRegion(moto.box, frame.width, frame.height)
      if (region.area > 1f && rules.wantsPlateCandidate(track.id, moto.box.area)) {
        rules.plateCandidate(track.id, moto.box.area, engine.crop(frame, region))
      }
    }

    // boxes are positions at capture time; the overlay moves them on by the time since then,
    // which also makes up for the time the vehicle model took
    onOverlay(OverlaySnapshot(frame.width, frame.height, capturedAtElapsedMs,
      tracker.live(wallMs).map { OverlayTrack(it.id, it.box, it.vx, it.vy, ridersOf[it.id] ?: emptyList()) }))

    finish(rules.collectFinished(wallMs) { queue.busy(it) })
  }

  // ---------------------------------------------------------------------------------------
  // [B] helmet worker

  private fun helmetLoop() {
    Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT)
    while (running) {
      val (key, job) = queue.take(200) ?: continue
      try {
        val t0 = System.nanoTime()
        val result = engine.classifyHelmet(job.frame, job.head)
        helmetNs.addAndGet(System.nanoTime() - t0)
        helmetLagNs.addAndGet(((System.currentTimeMillis() - job.atMs).coerceAtLeast(0L)) * 1_000_000L)
        helmetCount.incrementAndGet()
        val score = if (result?.label == "no_helmet") shotScore(job.rider.box.area, headSharpness(job)) else 0.0
        rules.vote(key, result, job.moto, score, job.atMs) {
          Frames(job.frame, engine.crop(job.frame, job.rider.box), job.riderBoxes)
        }
      } catch (e: Throwable) {
        Log.e("RSAFE", "Helmet check failed", e)
      } finally {
        queue.done(key)
      }
    }
  }

  private fun headSharpness(job: HelmetJob): Double {
    val w = job.head.width()
    val h = job.head.height()
    if (w < 3 || h < 3) return 0.0
    val px = IntArray(w * h)
    job.frame.getPixels(px, 0, w, job.head.left, job.head.top, w, h)
    return sharpness(px, w, h, step = if (w * h > 40_000) 2 else 1)
  }

  // ---------------------------------------------------------------------------------------
  // [C] evidence worker: plate from the saved candidates, then save + hash + hand over.

  private fun finish(reports: List<TrackReport<Frames, Bitmap>>) {
    for (report in reports) {
      evidence.execute {
        try {
          val t0 = SystemClock.elapsedRealtime()
          var best = PlateResult<Bitmap>("not_visible")
          for (crop in report.plateCandidates.take(min(s.rules.plateCandidates, s.rules.maxPlateAttempts))) {
            val r = engine.readPlate(crop)
            if (plateBetter(r, best)) best = r
            if (best.status == "read") break
          }
          lastPlateMs = SystemClock.elapsedRealtime() - t0
          engine.save(report, best)?.let(onViolation)
        } catch (e: Throwable) {
          Log.e("RSAFE", "Could not finish violation", e)
        }
      }
    }
  }

  /**
   * Hand over every motorcycle still open (duty ends / phone turned): first let the helmet
   * worker finish what is queued (max waitMs), then close all. Call on the camera thread.
   */
  fun flushAll(waitMs: Long = 1500) {
    val until = SystemClock.elapsedRealtime() + waitMs
    while (!queue.idle && SystemClock.elapsedRealtime() < until) Thread.sleep(20)
    finish(rules.flush())
    tracker.clear()
  }

  /** Stop the workers after the evidence of finished violations is written (max waitMs). */
  fun close(waitMs: Long = 20_000) {
    running = false
    queue.clear()
    evidence.shutdown()
    try {
      evidence.awaitTermination(waitMs, TimeUnit.MILLISECONDS)
    } catch (_: InterruptedException) {}
    helmetThread.join(1000)
  }
}
