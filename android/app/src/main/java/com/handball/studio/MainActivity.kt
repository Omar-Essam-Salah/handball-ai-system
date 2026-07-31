package com.handball.studio

import android.app.Activity
import android.net.wifi.WifiManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicReference

/**
 * Thin WebView shell for Handball Studio.
 *
 * It contains NO match UI of its own — that all lives in the web app served by
 * the PC (webapp/). The shell only finds the PC server (typed or auto-scanned
 * over WiFi) and loads it. So to change features/look you edit the web files on
 * the PC; this APK rarely needs rebuilding.
 */
class MainActivity : Activity() {

    private lateinit var web: WebView
    private lateinit var connectView: LinearLayout
    private lateinit var status: TextView
    private lateinit var input: EditText
    private val ui = Handler(Looper.getMainLooper())
    private val prefs by lazy { getSharedPreferences("hb", MODE_PRIVATE) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = FrameLayout(this)
        web = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.mediaPlaybackRequiresUserGesture = false
            @Suppress("DEPRECATION")
            settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            webViewClient = WebViewClient()
            webChromeClient = WebChromeClient()
            visibility = View.GONE
        }
        root.addView(web, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        root.addView(buildConnect(), FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        setContentView(root)

        val saved = prefs.getString("url", "") ?: ""
        if (saved.isNotEmpty()) {
            input.setText(saved.removePrefix("http://"))
            tryConnect(saved)
        }
    }

    private fun buildConnect(): View {
        connectView = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(70, 70, 70, 70)
            setBackgroundColor(0xFF0B0D11.toInt())
        }
        val title = TextView(this).apply {
            text = "🤾  Handball Studio"; textSize = 26f
            setTextColor(0xFFE8EAEF.toInt()); gravity = Gravity.CENTER
        }
        status = TextView(this).apply {
            text = "Enter your PC's address, or tap Auto-find."
            setTextColor(0xFF8A93A6.toInt()); gravity = Gravity.CENTER
            setPadding(0, 30, 0, 30)
        }
        input = EditText(this).apply {
            hint = "192.168.1.20:8000"
            setTextColor(0xFFE8EAEF.toInt()); setHintTextColor(0xFF5A6172.toInt())
        }
        val connect = Button(this).apply {
            text = "Connect"
            setOnClickListener { tryConnect(normalize(input.text.toString())) }
        }
        val auto = Button(this).apply {
            text = "Auto-find on WiFi"
            setOnClickListener { autoFind() }
        }
        connectView.addView(title)
        connectView.addView(status)
        connectView.addView(input)
        connectView.addView(connect)
        connectView.addView(auto)
        return connectView
    }

    private fun normalize(s: String): String {
        var u = s.trim()
        if (u.isEmpty()) return ""
        if (!u.startsWith("http")) u = "http://$u"
        val hostPart = u.removePrefix("http://").removePrefix("https://")
        if (!hostPart.contains(":")) u += ":8000"
        return u
    }

    private fun tryConnect(url: String) {
        if (url.isEmpty()) return
        status.text = "Connecting to $url …"
        Thread {
            val ok = ping(url)
            ui.post {
                if (ok) {
                    prefs.edit().putString("url", url).apply()
                    connectView.visibility = View.GONE
                    web.visibility = View.VISIBLE
                    web.loadUrl(url)
                } else {
                    status.text = "No server at $url.\nIs Handball_Mobile.bat running and on the same WiFi?"
                }
            }
        }.start()
    }

    private fun ping(base: String): Boolean = try {
        val c = URL("$base/ping").openConnection() as HttpURLConnection
        c.connectTimeout = 1200; c.readTimeout = 1200
        val body = c.inputStream.bufferedReader().use { it.readText() }
        c.disconnect()
        body.contains("\"app\"") && body.contains("handball")
    } catch (e: Exception) { false }

    private fun autoFind() {
        status.text = "Scanning WiFi for the server…"
        Thread {
            val base = localSubnet()
            if (base == null) {
                ui.post { status.text = "Not connected to WiFi?" }
                return@Thread
            }
            val found = AtomicReference<String?>(null)
            val workers = (1..254).chunked(26).map { chunk ->
                Thread {
                    for (i in chunk) {
                        if (found.get() != null) break
                        val u = "http://$base.$i:8000"
                        if (ping(u)) { found.compareAndSet(null, u); break }
                    }
                }
            }
            workers.forEach { it.start() }
            workers.forEach { it.join() }
            val hit = found.get()
            ui.post {
                if (hit != null) {
                    input.setText(hit.removePrefix("http://"))
                    tryConnect(hit)
                } else {
                    status.text = "No server found on $base.0/24.\nType the address shown on the PC."
                }
            }
        }.start()
    }

    private fun localSubnet(): String? = try {
        val wifi = applicationContext.getSystemService(WIFI_SERVICE) as WifiManager
        @Suppress("DEPRECATION")
        val ip = wifi.connectionInfo.ipAddress
        if (ip == 0) null
        else "${ip and 0xff}.${ip shr 8 and 0xff}.${ip shr 16 and 0xff}"
    } catch (e: Exception) { null }

    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        when {
            web.visibility == View.VISIBLE && web.canGoBack() -> web.goBack()
            web.visibility == View.VISIBLE -> {
                web.visibility = View.GONE
                connectView.visibility = View.VISIBLE
            }
            else -> super.onBackPressed()
        }
    }
}
