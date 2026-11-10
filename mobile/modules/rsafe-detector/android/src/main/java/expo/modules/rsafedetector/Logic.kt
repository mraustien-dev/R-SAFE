// Pure detection logic, ported from the laptop engine (ai-engine/rsafe) and extended for the
// phone's two-speed pipeline. No Android types here, so it runs in plain JVM unit tests.
package expo.modules.rsafedetector

import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float) {
  val width get() = x2 - x1
  val height get() = y2 - y1
  val area get() = max(0f, width) * max(0f, height)
  val cx get() = (x1 + x2) / 2f
  val cy get() = (y1 + y2) / 2f

  fun iou(o: Box): Float {
    val inter = intersection(o)
    val union = area + o.area - inter
    return if (union <= 0f) 0f else inter / union
  }

  fun intersection(o: Box): Float {
    val w = min(x2, o.x2) - max(x1, o.x1)
    val h = min(y2, o.y2) - max(y1, o.y1)
    return max(0f, w) * max(0f, h)
  }

  fun expand(ratio: Float): Box {
    val dx = width * ratio
    val dy = height * ratio
    return Box(x1 - dx, y1 - dy, x2 + dx, y2 + dy)
  }

  fun shift(dx: Float, dy: Float) = Box(x1 + dx, y1 + dy, x2 + dx, y2 + dy)

  fun contains(x: Float, y: Float) = x in x1..x2 && y in y1..y2

  fun toList() = listOf(x1.toDouble(), y1.toDouble(), x2.toDouble(), y2.toDouble())

  companion object {
    fun ofCentre(cx: Float, cy: Float, w: Float, h: Float) = Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
  }
}

data class Det(val box: Box, val conf: Float, val cls: Int, val trackId: Int = -1)

data class HelmetResult(val label: String, val conf: Float) // "helmet" | "no_helmet"

/** Class-aware greedy non-maximum suppression (same idea as Ultralytics / cv2.dnn.NMSBoxes). */
fun nms(dets: List<Det>, iouThreshold: Float): List<Det> {
  val kept = ArrayList<Det>()
  for ((_, group) in dets.groupBy { it.cls }) {
    val sorted = group.sortedByDescending { it.conf }.toMutableList()
    while (sorted.isNotEmpty()) {
      val best = sorted.removeAt(0)
      kept.add(best)
      sorted.removeAll { it.box.iou(best.box) > iouThreshold }
    }
  }
  return kept
}

// ---------------------------------------------------------------------------------------
// Riders (rsafe/association/riders.py)

/**
 * {motorcycle index -> riders}, riders ordered front (position 1) to back.
 * A person rides a motorcycle when the bottom-centre of their box is inside the motorcycle
 * box enlarged by expandRatio per side; ties go to the largest overlap; taller = further forward.
 */
fun associateRiders(
  motorcycles: List<Det>,
  persons: List<Det>,
  expandRatio: Float,
  maxRiders: Int,
): Map<Int, List<Det>> {
  val riders = HashMap<Int, MutableList<Det>>()
  motorcycles.indices.forEach { riders[it] = ArrayList() }
  val expanded = motorcycles.map { it.box.expand(expandRatio) }
  for (person in persons) {
    val px = person.box.cx
    val py = person.box.y2
    val candidates = expanded.indices.filter { expanded[it].contains(px, py) }
    if (candidates.isEmpty()) continue
    val best = candidates.maxByOrNull { person.box.intersection(motorcycles[it].box) }!!
    riders[best]!!.add(person)
  }
  return riders.mapValues { (_, list) ->
    list.sortedWith(compareBy({ -it.box.height }, { it.box.cx })).take(maxRiders)
  }
}

/** Integer crop rectangle (left, top, right, bottom) of a rider's head, clipped to the frame. */
fun headCropRect(
  rider: Box,
  headRatio: Float,
  padRatio: Float,
  frameW: Int,
  frameH: Int,
): IntArray? {
  val headH = rider.height * headRatio
  val padX = rider.width * padRatio
  val padY = headH * padRatio
  val left = max(0, (rider.x1 - padX).roundToInt())
  val top = max(0, (rider.y1 - padY).roundToInt())
  val right = min(frameW, (rider.x2 + padX).roundToInt())
  val bottom = min(frameH, (rider.y1 + headH + padY).roundToInt())
  if (right <= left || bottom <= top) return null
  return intArrayOf(left, top, right, bottom)
}

/** Motorcycle box widened 10% each side and extended 25% downward (rsafe/ocr/plate_reader.py). */
fun plateSearchRegion(moto: Box, frameW: Int, frameH: Int): Box {
  val padX = moto.width * 0.10f
  val padDown = moto.height * 0.25f
  return Box(
    max(0f, moto.x1 - padX),
    max(0f, moto.y1),
    min(frameW.toFloat(), moto.x2 + padX),
    min(frameH.toFloat(), moto.y2 + padDown),
  )
}

fun normalizePlateText(raw: String) = raw.uppercase().replace(Regex("[^A-Z0-9]"), "")

// ---------------------------------------------------------------------------------------
// Image measures (ARGB pixels, as from Bitmap.getPixels)

/**
 * Sharpness = variance of the Laplacian of the grey image (standard blur measure: blurred
 * crops have weak edges, so a low value). Samples every `step`-th pixel for speed.
 */
fun sharpness(pixels: IntArray, w: Int, h: Int, step: Int = 1): Double {
  if (w < 3 || h < 3) return 0.0
  fun grey(x: Int, y: Int): Int {
    val p = pixels[y * w + x]
    return (((p shr 16) and 0xFF) * 77 + ((p shr 8) and 0xFF) * 150 + (p and 0xFF) * 29) shr 8
  }
  var sum = 0.0
  var sumSq = 0.0
  var n = 0
  var y = 1
  while (y < h - 1) {
    var x = 1
    while (x < w - 1) {
      val lap = (grey(x - 1, y) + grey(x + 1, y) + grey(x, y - 1) + grey(x, y + 1) - 4 * grey(x, y)).toDouble()
      sum += lap
      sumSq += lap * lap
      n++
      x += step
    }
    y += step
  }
  if (n == 0) return 0.0
  val mean = sum / n
  return sumSq / n - mean * mean
}

/** Mean brightness 0..255 of every `step`-th pixel. */
fun meanLuma(pixels: IntArray, step: Int = 1): Double {
  var sum = 0L
  var n = 0
  var i = 0
  while (i < pixels.size) {
    val p = pixels[i]
    sum += (((p shr 16) and 0xFF) * 77 + ((p shr 8) and 0xFF) * 150 + (p and 0xFF) * 29) shr 8
    n++
    i += step
  }
  return if (n == 0) 0.0 else sum.toDouble() / n
}

/**
 * How good a moment is for evidence: a big, sharp rider beats a small or blurred one.
 * Sharpness counts fully up to `sharpRef` (a clearly sharp crop), then stops adding.
 */
fun shotScore(riderArea: Float, sharp: Double, sharpRef: Double = 120.0): Double =
  riderArea * (0.25 + 0.75 * min(1.0, sharp / sharpRef))

// ---------------------------------------------------------------------------------------
// Motion-priority exposure. Short shutter first (sharp fast riders), sensitivity (ISO) rises
// before the shutter is allowed to get longer. Dead band around the target stops flicker.

data class Exposure(val shutterNs: Long, val iso: Int)

class ExposureController(
  isoRange: IntRange,
  shutterRangeNs: LongRange,
  private val target: Double = 115.0,
  private val deadBand: Double = 18.0,
  maxIso: Int = 3200,
) {
  /** Shutter steps from fastest to slowest: 1/1000, 1/500 (preferred), 1/250, 1/125, 1/60 s. */
  private val ladder = listOf(1_000_000L, 2_000_000L, 4_000_000L, 8_000_000L, 16_666_667L)
    .filter { it in shutterRangeNs }.ifEmpty { listOf(shutterRangeNs.first) }
  private val isoMin = isoRange.first
  private val isoMax = min(isoRange.last, max(isoRange.first, maxIso))
  private var step = ladder.indexOf(2_000_000L).let { if (it < 0) 0 else it }
  private var iso = min(isoMax, max(isoMin, 400))

  val current get() = Exposure(ladder[step], iso)

  /** Next exposure for a measured mean brightness (0..255); same as current inside the dead band. */
  fun update(meanLuma: Double): Exposure {
    if (abs(meanLuma - target) <= deadBand) return current
    // light needed relative to now, limited per step so changes are gradual
    val ratio = (target / max(meanLuma, 1.0)).coerceIn(0.6, 1.6)
    if (ratio > 1) { // too dark: ISO first, then a longer shutter
      if (iso < isoMax) {
        iso = min(isoMax, (iso * ratio).roundToInt())
      } else if (step < ladder.size - 1) {
        step++
        iso = max(isoMin, (iso / 2.0).roundToInt()) // shutter doubled: halve ISO to compensate
      }
    } else { // too bright: shorter shutter back toward 1/500 first, then lower ISO
      val preferred = ladder.indexOf(2_000_000L).let { if (it < 0) 0 else it }
      if (step > preferred) {
        step--
        iso = min(isoMax, (iso * 2.0).roundToInt())
      } else if (iso > isoMin) {
        iso = max(isoMin, (iso * ratio).roundToInt())
      } else if (step > 0) {
        step--
      }
    }
    return current
  }
}

// ---------------------------------------------------------------------------------------
// Motorcycle tracking: ByteTrack-style (like the laptop engine) with motion prediction.
//  * each track predicts where it is now (position + velocity), matching uses that box, so a
//    fast motorcycle that moved further than its own width keeps its id
//  * pass 1 matches confident detections, pass 2 lets weak ones (blur) only CONTINUE tracks
//  * a track must be seen twice before it counts (no one-frame ghosts)

data class TrackerConfig(
  val highConf: Float = 0.35f,
  val lowConf: Float = 0.10f,
  val minIou: Float = 0.10f,
  val maxCentreJump: Float = 0.9f, // of the predicted box's larger side
  val firstJump: Float = 2.0f,     // same, for a track seen once (no speed known yet)
  val confirmHits: Int = 2,
  val lostMs: Long = 1500,
  val tentativeLostMs: Long = 400,
  val posGain: Float = 0.7f, // how much a detection corrects the predicted position
  val velGain: Float = 0.4f, // how much it corrects the velocity
)

/** A track as seen by the rest of the app. Velocity in pixels per millisecond. */
data class TrackState(val id: Int, val det: Det, val box: Box, val vx: Float, val vy: Float, val atMs: Long) {
  fun predict(nowMs: Long): Box {
    val dt = (nowMs - atMs).coerceIn(0L, 500L).toFloat()
    return box.shift(vx * dt, vy * dt)
  }
}

class MotoTracker(private val cfg: TrackerConfig = TrackerConfig()) {
  private class Track(val id: Int, var box: Box, var det: Det, var atMs: Long) {
    var vx = 0f
    var vy = 0f
    var hits = 1
    var lastSeenMs = atMs

    fun predicted(nowMs: Long): Box {
      val dt = (nowMs - atMs).coerceIn(0L, 500L).toFloat()
      return box.shift(vx * dt, vy * dt)
    }
  }

  private val tracks = ArrayList<Track>()
  private var nextId = 1

  /**
   * Feed one frame's motorcycle detections (all of them, weak ones included).
   * Returns the CONFIRMED tracks seen in this frame, each with the detection that matched it.
   */
  fun update(dets: List<Det>, nowMs: Long): List<TrackState> {
    tracks.removeAll { t ->
      val limit = if (t.hits >= cfg.confirmHits) cfg.lostMs else cfg.tentativeLostMs
      nowMs - t.lastSeenMs > limit
    }
    val high = dets.filter { it.conf >= cfg.highConf }
    val low = dets.filter { it.conf >= cfg.lowConf && it.conf < cfg.highConf }

    val matched = HashMap<Track, Det>()
    val unmatchedHigh = match(high, tracks, nowMs, matched)
    match(low, tracks.filter { it !in matched && it.hits >= cfg.confirmHits }, nowMs, matched)

    for ((t, d) in matched) correct(t, d, nowMs)
    for (d in unmatchedHigh) tracks.add(Track(nextId++, d.box, d, nowMs))

    return tracks.filter { it.lastSeenMs == nowMs && it.hits >= cfg.confirmHits }
      .map { TrackState(it.id, it.det.copy(trackId = it.id), it.box, it.vx, it.vy, it.atMs) }
  }

  /** Every confirmed track still alive, at its predicted position for `nowMs` (for drawing). */
  fun live(nowMs: Long): List<TrackState> = tracks.filter { it.hits >= cfg.confirmHits }
    .map { TrackState(it.id, it.det.copy(trackId = it.id), it.predicted(nowMs), it.vx, it.vy, nowMs) }

  val size get() = tracks.count { it.hits >= cfg.confirmHits }

  /**
   * Forget all tracks (e.g. the phone turned, so every box moved) but keep counting ids up:
   * a reused id would collide with a motorcycle the rules have already closed.
   */
  fun clear() = tracks.clear()

  /** Greedy best-score matching against predicted boxes. Returns the detections left over. */
  private fun match(dets: List<Det>, pool: List<Track>, nowMs: Long, matched: HashMap<Track, Det>): List<Det> {
    val free = pool.filter { it !in matched }
    val pairs = ArrayList<Triple<Int, Track, Float>>()
    for ((i, d) in dets.withIndex()) {
      for (t in free) {
        val p = t.predicted(nowMs)
        val iou = d.box.iou(p)
        val dx = d.box.cx - p.cx
        val dy = d.box.cy - p.cy
        val jump = if (t.hits == 1) cfg.firstJump else cfg.maxCentreJump
        val reach = jump * max(max(p.width, p.height), 1f)
        val dist = sqrt(dx * dx + dy * dy)
        if (iou >= cfg.minIou || dist <= reach) {
          pairs.add(Triple(i, t, iou + (1f - min(dist / reach, 1f)) * 0.5f))
        }
      }
    }
    pairs.sortByDescending { it.third }
    val usedDets = HashSet<Int>()
    for ((i, t, _) in pairs) {
      if (i in usedDets || t in matched) continue
      matched[t] = dets[i]
      usedDets.add(i)
    }
    return dets.filterIndexed { i, _ -> i !in usedDets }
  }

  /** Alpha-beta filter: blend the prediction with the detection, update the velocity. */
  private fun correct(t: Track, d: Det, nowMs: Long) {
    val dt = (nowMs - t.atMs).coerceAtLeast(1L).toFloat()
    val cx: Float
    val cy: Float
    if (t.hits == 1) {
      // second sighting: the first real speed measurement, take it as is
      t.vx = (d.box.cx - t.box.cx) / dt
      t.vy = (d.box.cy - t.box.cy) / dt
      cx = d.box.cx
      cy = d.box.cy
    } else {
      val p = t.predicted(nowMs)
      val rx = d.box.cx - p.cx
      val ry = d.box.cy - p.cy
      cx = p.cx + cfg.posGain * rx
      cy = p.cy + cfg.posGain * ry
      t.vx += cfg.velGain * rx / dt
      t.vy += cfg.velGain * ry / dt
    }
    // size follows the detection directly (it changes slowly and is measured well)
    t.box = Box.ofCentre(cx, cy, d.box.width, d.box.height)
    t.det = d
    t.atMs = nowMs
    t.lastSeenMs = nowMs
    t.hits++
  }
}

// ---------------------------------------------------------------------------------------
// Helmet jobs: the tracker loop offers head crops, the helmet worker takes them.
// One pending job per rider: a newer crop REPLACES the waiting one, so the worker never
// builds a backlog and always classifies the freshest view of each rider.

data class RiderKey(val trackId: Int, val position: Int)

class HelmetQueue<J> {
  private val lock = ReentrantLock()
  private val notEmpty = lock.newCondition()
  private val pending = LinkedHashMap<RiderKey, J>()
  private val running = HashMap<Int, Int>() // trackId -> jobs being classified now

  fun offer(key: RiderKey, job: J) = lock.withLock {
    pending.remove(key) // re-insert at the back: fair turn-taking between riders
    pending[key] = job
    notEmpty.signal()
  }

  /** Next job, waiting up to timeoutMs; null if none. Call done() when finished. */
  fun take(timeoutMs: Long): Pair<RiderKey, J>? = lock.withLock {
    var waitNs = TimeUnit.MILLISECONDS.toNanos(timeoutMs)
    while (pending.isEmpty()) {
      if (waitNs <= 0L) return null
      waitNs = notEmpty.awaitNanos(waitNs)
    }
    val it = pending.entries.iterator()
    val e = it.next()
    it.remove()
    running[e.key.trackId] = (running[e.key.trackId] ?: 0) + 1
    e.key to e.value
  }

  fun done(key: RiderKey) = lock.withLock {
    val n = (running[key.trackId] ?: 1) - 1
    if (n <= 0) running.remove(key.trackId) else running[key.trackId] = n
  }

  /** True while a rider of this motorcycle is waiting or being classified. */
  fun busy(trackId: Int): Boolean = lock.withLock {
    trackId in running || pending.keys.any { it.trackId == trackId }
  }

  val size get() = lock.withLock { pending.size }

  /** Nothing waiting and nothing being classified. */
  val idle get() = lock.withLock { pending.isEmpty() && running.isEmpty() }

  fun clear() = lock.withLock { pending.clear() }
}

// ---------------------------------------------------------------------------------------
// Violation rules (rsafe/violations/rules.py), fed asynchronously:
//  * seen()  - tracker loop, every frame the motorcycle is tracked
//  * vote()  - helmet worker, when a head crop has been classified
//  * plateCandidate() - tracker loop, keeps the K frames where the motorcycle was largest
//  * collectFinished() closes a motorcycle lost for trackLostMs AND with no helmet check
//    still queued for it, so no vote is lost. Thread-safe.
// E = evidence payload (frame etc.), P = plate candidate payload (crop).

data class RulesConfig(
  val window: Int = 10,
  val minConfidence: Float = 0.50f,
  val minVotes: Int = 3,
  val trackLostMs: Long = 1500,
  val maxPlateAttempts: Int = 5,
  val plateCandidates: Int = 3,
)

data class PlateResult<P>(
  val status: String, // "read" | "unreadable" | "not_visible"
  val text: String? = null,
  val conf: Float? = null,
  val crop: P? = null,
)

private val PLATE_RANK = mapOf("not_visible" to 0, "unreadable" to 1, "read" to 2)

/** True when `new` is a better plate reading than `old`. */
fun <P> plateBetter(new: PlateResult<P>, old: PlateResult<P>): Boolean {
  val a = PLATE_RANK.getValue(new.status)
  val b = PLATE_RANK.getValue(old.status)
  if (a != b) return a > b
  return new.status == "read" && (new.conf ?: 0f) > (old.conf ?: 0f)
}

data class RiderVerdict(val position: Int, val helmet: String, val confidence: Float, val framesObserved: Int)

class TrackReport<E, P>(
  val trackId: Int,
  val riders: List<RiderVerdict>,
  val motorcycle: Det,
  val evidence: E,
  val plateCandidates: List<P>, // largest motorcycle views first
  val seenAtMs: Long,
)

class ViolationTracker<E, P>(private val rules: RulesConfig) {
  private class Shot<E>(val score: Double, val payload: E, val moto: Det, val atMs: Long)

  private class State<E, P>(var lastSeenMs: Long, var moto: Det) {
    val votes = HashMap<Int, ArrayDeque<HelmetResult>>()
    val last = HashMap<Int, HelmetResult>() // latest confident result per rider (overlay)
    var shot: Shot<E>? = null
    val plates = ArrayList<Pair<Float, P>>() // (motorcycle area, crop), largest first
  }

  private val tracks = HashMap<Int, State<E, P>>()
  private val closed = HashSet<Int>()

  @get:Synchronized
  val openTracks get() = tracks.size

  @Synchronized
  fun seen(moto: Det, nowMs: Long) {
    if (moto.trackId in closed) return
    val t = tracks.getOrPut(moto.trackId) { State(nowMs, moto) }
    t.lastSeenMs = nowMs
    t.moto = moto
  }

  /**
   * A classified head crop. When it is a confident no_helmet vote and this moment scores
   * higher than the stored one, makeEvidence() is called and its result kept as the photo.
   */
  @Synchronized
  fun vote(key: RiderKey, result: HelmetResult?, moto: Det, score: Double, atMs: Long, makeEvidence: () -> E) {
    val t = tracks[key.trackId] ?: return // already closed (or never seen)
    if (result == null || result.conf < rules.minConfidence) return
    val q = t.votes.getOrPut(key.position) { ArrayDeque() }
    q.addLast(result)
    while (q.size > rules.window) q.removeFirst()
    t.last[key.position] = result
    if (result.label == "no_helmet") {
      val s = t.shot
      if (s == null || score > s.score) t.shot = Shot(score, makeEvidence(), moto, atMs)
    }
  }

  /** Does this motorcycle view belong among the K largest kept for plate reading? */
  @Synchronized
  fun wantsPlateCandidate(trackId: Int, area: Float): Boolean {
    val t = tracks[trackId] ?: return false
    return t.plates.size < rules.plateCandidates || area > t.plates.last().first
  }

  @Synchronized
  fun plateCandidate(trackId: Int, area: Float, crop: P) {
    val t = tracks[trackId] ?: return
    t.plates.add(area to crop)
    t.plates.sortByDescending { it.first }
    while (t.plates.size > rules.plateCandidates) t.plates.removeAt(t.plates.size - 1)
  }

  /** Latest confident helmet result per rider position (for the overlay). */
  @Synchronized
  fun lastResults(trackId: Int): Map<Int, HelmetResult> = tracks[trackId]?.last?.toMap() ?: emptyMap()

  /** Motorcycles lost for trackLostMs with no helmet check pending. */
  @Synchronized
  fun collectFinished(nowMs: Long, busy: (Int) -> Boolean = { false }): List<TrackReport<E, P>> =
    close(tracks.filter { (id, t) -> nowMs - t.lastSeenMs > rules.trackLostMs && !busy(id) }.keys.toList())

  @Synchronized
  fun flush(): List<TrackReport<E, P>> = close(tracks.keys.toList())

  private fun close(ids: List<Int>): List<TrackReport<E, P>> {
    val out = ArrayList<TrackReport<E, P>>()
    for (id in ids) {
      val t = tracks.remove(id) ?: continue
      closed.add(id)
      report(id, t)?.let { out.add(it) }
    }
    return out
  }

  private fun verdict(position: Int, votes: ArrayDeque<HelmetResult>): RiderVerdict {
    val no = votes.filter { it.label == "no_helmet" }.map { it.conf }
    val yes = votes.filter { it.label == "helmet" }.map { it.conf }
    val (label, winning) = when {
      no.size >= rules.minVotes && no.size > yes.size -> "no_helmet" to no
      yes.size > no.size -> "helmet" to yes
      else -> "unknown" to emptyList()
    }
    val conf = if (winning.isEmpty()) 0f else (winning.average() * 10000).roundToInt() / 10000f
    return RiderVerdict(position, label, conf, votes.size)
  }

  private fun report(id: Int, t: State<E, P>): TrackReport<E, P>? {
    val riders = t.votes.keys.sorted().map { verdict(it, t.votes.getValue(it)) }
    val shot = t.shot ?: return null
    if (riders.none { it.helmet == "no_helmet" }) return null
    return TrackReport(id, riders, shot.moto, shot.payload, t.plates.map { it.second }, shot.atMs)
  }
}
