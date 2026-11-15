package expo.modules.rsafedetector

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.io.File

class RsafeDetectorModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("RsafeDetector")

    // SHA-256 of a file on disk (used to re-check evidence before upload)
    AsyncFunction("fileSha256") { path: String ->
      sha256Hex(File(path.removePrefix("file://")).readBytes())
    }

    // Removes an evidence folder ONLY after the server confirmed it (called by the sync code).
    AsyncFunction("deleteUploadedEvidence") { folder: String ->
      val root = File(appContext.reactContext!!.filesDir, "evidence").canonicalFile
      val dir = File(folder.removePrefix("file://")).canonicalFile
      if (dir.parentFile == root) dir.deleteRecursively() else false
    }

    // Every complete evidence folder still on the phone, as event JSON text.
    AsyncFunction("listEvidence") {
      val root = File(appContext.reactContext!!.filesDir, "evidence")
      (root.listFiles() ?: emptyArray())
        .mapNotNull { File(it, "event.json").takeIf { f -> f.isFile }?.readText() }
    }

    View(RsafeCameraView::class) {
      Events("onStatus", "onViolation", "onError")

      Prop("active") { view: RsafeCameraView, active: Boolean ->
        view.setActive(active)
      }

      OnViewDestroys { view: RsafeCameraView ->
        view.release()
      }
    }
  }
}
