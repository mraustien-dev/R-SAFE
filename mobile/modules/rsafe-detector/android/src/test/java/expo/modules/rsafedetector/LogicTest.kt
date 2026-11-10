package expo.modules.rsafedetector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.concurrent.thread

class LogicTest {
  private fun det(x1: Float, y1: Float, x2: Float, y2: Float, conf: Float = 0.9f, cls: Int = 0, id: Int = -1) =
    Det(Box(x1, y1, x2, y2), conf, cls, id)

  private fun moto(cx: Float, cy: Float = 500f, size: Float = 120f, conf: Float = 0.8f) =
    Det(Box.ofCentre(cx, cy, size, size), conf, 3)

  // ---- geometry / riders ------------------------------------------------------------------

  @Test fun nmsKeepsBestPerClass() {
    val kept = nms(
      listOf(det(0f, 0f, 10f, 10f, 0.9f), det(1f, 1f, 11f, 11f, 0.8f), det(1f, 1f, 11f, 11f, 0.7f, cls = 1)),
      0.45f,
    )
    assertEquals(2, kept.size)
    assertTrue(kept.any { it.cls == 0 && it.conf == 0.9f })
  }

  @Test fun ridersMatchedByBottomCentreAndOrderedTallestFirst() {
    val moto = det(100f, 100f, 300f, 300f, cls = 3)
    val front = det(150f, 20f, 250f, 280f)
    val back = det(160f, 60f, 240f, 270f)
    val walker = det(500f, 50f, 560f, 280f)
    val r = associateRiders(listOf(moto), listOf(back, walker, front), 0.15f, 3)
    assertEquals(listOf(front, back), r[0])
  }

  @Test fun headCropIsTopOfRiderClippedToFrame() {
    val rect = headCropRect(Box(10f, 0f, 50f, 100f), 0.4f, 0.2f, 40, 200)!!
    assertEquals(listOf(2, 0, 40, 48), rect.toList())
  }

  @Test fun plateTextNormalised() = assertEquals("ABC1234", normalizePlateText("abc-12 34"))

  // ---- tracker -----------------------------------------------------------------------------

  /** Frames every `stepMs`, motorcycle moving `pxPerFrame` to the right. Returns ids per frame. */
  private fun crossing(tracker: MotoTracker, pxPerFrame: Float, frames: Int, stepMs: Long = 100, startX: Float = 60f) =
    (0 until frames).map { i -> tracker.update(listOf(moto(startX + i * pxPerFrame)), i * stepMs).map { it.id } }

  @Test fun fastMotorcycleKeepsOneIdAcrossTheFrame() {
    // 1920 px in ~1 s at 10 fps: moves 190 px per frame, far more than its 120 px width
    val ids = crossing(MotoTracker(), 190f, 10).drop(1) // first frame: not confirmed yet
    assertTrue(ids.all { it.size == 1 })
    assertEquals(1, ids.flatten().toSet().size)
  }

  @Test fun oneFrameGhostNeverBecomesATrack() {
    val t = MotoTracker()
    assertTrue(t.update(listOf(moto(500f)), 0).isEmpty())   // seen once: tentative
    assertTrue(t.update(emptyList(), 100).isEmpty())
    assertTrue(t.update(emptyList(), 600).isEmpty())         // tentative dropped after 400 ms
    assertEquals(0, t.size)
  }

  @Test fun twoMotorcyclesPassingKeepTheirIds() {
    val t = MotoTracker()
    var a = -1
    var b = -1
    for (i in 0 until 12) {
      val left = moto(200f + i * 80f, cy = 400f)    // going right
      val right = moto(1400f - i * 80f, cy = 650f)  // going left, other lane
      val out = t.update(listOf(left, right), i * 100L)
      if (i == 1) {
        a = out.first { it.det.box.cy < 500f }.id
        b = out.first { it.det.box.cy > 500f }.id
      }
      if (i >= 1) {
        assertEquals(a, out.first { it.det.box.cy < 500f }.id)
        assertEquals(b, out.first { it.det.box.cy > 500f }.id)
      }
    }
    assertNotEquals(a, b)
  }

  @Test fun shortlyHiddenMotorcycleIsPickedUpAgain() {
    val t = MotoTracker()
    t.update(listOf(moto(100f)), 0)
    val id = t.update(listOf(moto(160f)), 100).single().id
    t.update(listOf(moto(220f)), 200)
    // hidden for 500 ms (behind a car), keeps moving 60 px / 100 ms
    val back = t.update(listOf(moto(580f)), 800).single().id
    assertEquals(id, back)
  }

  @Test fun weakDetectionContinuesButNeverStartsATrack() {
    val t = MotoTracker()
    t.update(listOf(moto(100f)), 0)
    val id = t.update(listOf(moto(150f)), 100).single().id
    assertEquals(id, t.update(listOf(moto(200f, conf = 0.15f)), 200).single().id) // blurred: still tracked
    val fresh = MotoTracker()
    repeat(5) { fresh.update(listOf(moto(500f, conf = 0.15f)), it * 100L) }
    assertEquals(0, fresh.size)
  }

  @Test fun predictionMovesAheadOfLastDetection() {
    val t = MotoTracker()
    repeat(5) { t.update(listOf(moto(100f + it * 50f)), it * 100L) } // 0.5 px/ms
    val now = t.live(450).single()
    val last = 100f + 4 * 50f
    assertTrue("predicted ${now.box.cx} should be ahead of $last", now.box.cx > last + 10f)
  }

  @Test fun clearKeepsCountingIdsSoClosedIdsAreNeverReused() {
    val t = MotoTracker()
    t.update(listOf(moto(100f)), 0)
    val a = t.update(listOf(moto(100f)), 100).single().id
    t.clear()
    t.update(listOf(moto(100f)), 200)
    val b = t.update(listOf(moto(100f)), 300).single().id
    assertTrue(b > a)
  }

  @Test fun queueIdleOnlyWhenNothingWaitingOrRunning() {
    val q = HelmetQueue<String>()
    assertTrue(q.idle)
    q.offer(RiderKey(1, 1), "x")
    assertFalse(q.idle)
    val (k, _) = q.take(0)!!
    assertFalse(q.idle)
    q.done(k)
    assertTrue(q.idle)
  }

  @Test fun lostTrackForgottenAfterLostMs() {
    val t = MotoTracker()
    t.update(listOf(moto(100f)), 0)
    val a = t.update(listOf(moto(100f)), 100).single().id
    t.update(emptyList(), 2000)
    t.update(listOf(moto(100f)), 2100)
    val b = t.update(listOf(moto(100f)), 2200).single().id
    assertNotEquals(a, b)
  }

  // ---- helmet queue ------------------------------------------------------------------------

  @Test fun newerCropReplacesWaitingOneForSameRider() {
    val q = HelmetQueue<String>()
    q.offer(RiderKey(1, 1), "old")
    q.offer(RiderKey(1, 1), "new")
    q.offer(RiderKey(2, 1), "other")
    assertEquals(2, q.size)
    val (k, job) = q.take(0)!!
    assertEquals(RiderKey(1, 1), k)
    assertEquals("new", job)
  }

  @Test fun busyUntilDoneSoTrackIsNotClosedEarly() {
    val q = HelmetQueue<String>()
    val key = RiderKey(7, 1)
    q.offer(key, "crop")
    assertTrue(q.busy(7))
    q.take(0)
    assertTrue(q.busy(7))  // being classified
    q.done(key)
    assertFalse(q.busy(7))
    assertNull(q.take(0))
  }

  @Test fun takeWaitsForAnOffer() {
    val q = HelmetQueue<String>()
    val th = thread { Thread.sleep(50); q.offer(RiderKey(1, 1), "late") }
    assertEquals("late", q.take(2000)?.second)
    th.join()
  }

  // ---- violation rules ---------------------------------------------------------------------

  private val m7 = det(0f, 0f, 100f, 100f, cls = 3, id = 7)

  private fun ViolationTracker<String, String>.votes(n: Int, label: String, conf: Float = 0.8f, t0: Long = 0) {
    seen(m7, t0)
    repeat(n) { vote(RiderKey(7, 1), HelmetResult(label, conf), m7, 1.0, t0 + it) { "frame$it" } }
  }

  @Test fun violationNeedsThreeVotesAndIsReportedOnce() {
    val v = ViolationTracker<String, String>(RulesConfig(trackLostMs = 1000))
    v.votes(3, "no_helmet")
    assertTrue(v.collectFinished(500).isEmpty())          // still in view
    val reports = v.collectFinished(2000)
    assertEquals(1, reports.size)
    assertEquals("no_helmet", reports[0].riders[0].helmet)
    v.votes(3, "no_helmet", t0 = 2100)                    // same track id again: ignored
    assertTrue(v.flush().isEmpty())
  }

  @Test fun notClosedWhileHelmetCheckPending() {
    val v = ViolationTracker<String, String>(RulesConfig(trackLostMs = 1000))
    v.votes(2, "no_helmet")
    assertTrue(v.collectFinished(5000) { it == 7 }.isEmpty()) // lost, but a check is queued
    v.vote(RiderKey(7, 1), HelmetResult("no_helmet", 0.9f), m7, 1.0, 5001) { "late" } // its vote
    assertEquals(1, v.collectFinished(5002) { false }.size)
  }

  @Test fun twoNoHelmetVotesAreNotEnough() {
    val v = ViolationTracker<String, String>(RulesConfig())
    v.votes(2, "no_helmet")
    assertTrue(v.flush().isEmpty())
  }

  @Test fun lowConfidenceVotesIgnored() {
    val v = ViolationTracker<String, String>(RulesConfig())
    v.votes(5, "no_helmet", conf = 0.4f)
    assertTrue(v.flush().isEmpty())
  }

  @Test fun noEvidenceWithoutViolator() {
    val v = ViolationTracker<String, String>(RulesConfig())
    v.votes(5, "helmet")
    assertNull(v.flush().firstOrNull())
  }

  @Test fun bestShotKeptAndMadeOnlyWhenBetter() {
    val v = ViolationTracker<String, String>(RulesConfig())
    v.seen(m7, 0)
    var made = 0
    for ((i, score) in listOf(1.0, 5.0, 3.0).withIndex()) {
      v.vote(RiderKey(7, 1), HelmetResult("no_helmet", 0.9f), m7, score, i.toLong()) { made++; "shot$score" }
    }
    assertEquals(2, made)                       // 1.0 then 5.0; 3.0 is worse, never built
    assertEquals("shot5.0", v.flush().single().evidence)
  }

  @Test fun plateCandidatesKeepTheLargestViews() {
    val v = ViolationTracker<String, String>(RulesConfig(plateCandidates = 2))
    v.votes(3, "no_helmet")
    for (area in listOf(10f, 30f, 20f, 5f)) {
      if (v.wantsPlateCandidate(7, area)) v.plateCandidate(7, area, "crop$area")
    }
    assertFalse(v.wantsPlateCandidate(7, 1f))
    assertEquals(listOf("crop30.0", "crop20.0"), v.flush().single().plateCandidates)
  }

  @Test fun betterPlateRanking() {
    assertTrue(plateBetter(PlateResult<String>("read", "A", 0.9f), PlateResult("unreadable")))
    assertFalse(plateBetter(PlateResult<String>("unreadable"), PlateResult("read", "A", 0.5f)))
    assertTrue(plateBetter(PlateResult<String>("read", "B", 0.95f), PlateResult("read", "A", 0.9f)))
  }

  // ---- image measures / exposure -----------------------------------------------------------

  @Test fun sharpEdgesScoreHigherThanFlat() {
    val w = 32
    val flat = IntArray(w * w) { 0xFF808080.toInt() }
    val stripes = IntArray(w * w) { i -> if ((i % w) % 2 == 0) 0xFFFFFFFF.toInt() else 0xFF000000.toInt() }
    assertEquals(0.0, sharpness(flat, w, w), 1e-9)
    assertTrue(sharpness(stripes, w, w) > 1000)
    assertTrue(shotScore(1000f, 200.0) > shotScore(1000f, 10.0))
    assertTrue(shotScore(4000f, 10.0) > shotScore(1000f, 200.0))
  }

  @Test fun meanLumaOfGrey() = assertEquals(128.0, meanLuma(IntArray(100) { 0xFF808080.toInt() }), 1.0)

  @Test fun exposurePrefersShortShutterAndRaisesIsoFirst() {
    val c = ExposureController(isoRange = 50..6400, shutterRangeNs = 100_000L..500_000_000L)
    assertEquals(2_000_000L, c.current.shutterNs) // starts at 1/500 s
    val start = c.current.iso
    c.update(60.0)                                 // too dark
    assertEquals(2_000_000L, c.current.shutterNs)  // shutter unchanged
    assertTrue(c.current.iso > start)
    repeat(30) { c.update(20.0) }                  // very dark: ISO maxes out, then a longer shutter
    assertTrue(c.current.shutterNs > 2_000_000L)
    assertTrue(c.current.iso <= 3200)              // ISO capped (noise)
  }

  @Test fun exposureStableInsideDeadBand() {
    val c = ExposureController(isoRange = 50..6400, shutterRangeNs = 100_000L..500_000_000L)
    val e = c.current
    repeat(10) { c.update(115.0 + (it % 3 - 1) * 10) } // small wobble around the target
    assertEquals(e, c.current)
  }

  @Test fun exposureBrightDaylightGoesBackToFastShutter() {
    val c = ExposureController(isoRange = 50..6400, shutterRangeNs = 100_000L..500_000_000L)
    repeat(30) { c.update(20.0) }  // dark: slow shutter
    repeat(40) { c.update(250.0) } // bright again
    assertTrue(c.current.shutterNs <= 2_000_000L)
  }
}
