"""
browser.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Browser factory for the scraper. Creates a configured headless
    Chrome instance with a residential proxy (via a local TCP
    forwarder that injects Proxy-Authorization), a fresh temp
    profile per call, fixed user agent / viewport / locale / timezone,
    anti-detection flags, and Chrome DevTools Protocol enabled for
    JSON interception. Every scrape begins from a clean slate.

Inputs:
    01_config/settings.py             USER_AGENT, viewport, locale,
                                      PROXY_URL, CHROMEDRIVER_PATH
    (none — library module)

Outputs:
    (none — library module; returns a Selenium WebDriver instance.
    Imported by the scraper orchestrators (run_*.py).)

Usage:
    from browser import create_browser, cleanup_browser
    driver = create_browser(headless=True, enable_cdp=True)
    try:
        ...
    finally:
        cleanup_browser(driver)
"""

import base64
import json
import logging
import os
import socket
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from settings_loader import load_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local proxy forwarder
# ---------------------------------------------------------------------------
# Chromium 145 broke MV2 extension APIs (chrome.proxy.settings.set and
# webRequest.onAuthRequired no longer function in headless mode).  Instead
# of relying on a Chrome extension for proxy auth, we run a tiny local TCP
# forwarder that:
#   1. Listens on 127.0.0.1:<free-port>  (no auth required)
#   2. Forwards every CONNECT / plain-HTTP request to the upstream Evomi
#      proxy, injecting a Proxy-Authorization header automatically.
# Chrome is pointed at localhost via --proxy-server, so it never needs to
# handle auth itself.
# ---------------------------------------------------------------------------

class _ProxyForwarder(threading.Thread):
    """Lightweight local-to-upstream proxy forwarder with auth injection."""

    def __init__(self, upstream_host: str, upstream_port: int,
                 username: str, password: str):
        super().__init__(daemon=True)
        self._upstream = (upstream_host, upstream_port)
        self._auth = base64.b64encode(
            f"{username}:{password}".encode()
        ).decode()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self.port = self._server.getsockname()[1]
        self._server.listen(32)
        self._running = True

    # -- public API ----------------------------------------------------------

    def stop(self):
        self._running = False
        try:
            self._server.close()
        except OSError:
            pass

    # -- internals -----------------------------------------------------------

    def run(self):
        self._server.settimeout(1.0)
        while self._running:
            try:
                client, _ = self._server.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._handle, args=(client,),
                             daemon=True).start()

    def _handle(self, client: socket.socket):
        upstream = None
        try:
            data = client.recv(65536)
            if not data:
                client.close()
                return

            upstream = socket.create_connection(self._upstream, timeout=15)
            auth_line = f"Proxy-Authorization: Basic {self._auth}\r\n".encode()

            if data[:8].upper().startswith(b"CONNECT "):
                # HTTPS tunnel: forward CONNECT with auth to upstream,
                # relay the 200 back to Chrome, then pipe raw bytes.
                header_end = data.find(b"\r\n\r\n")
                # Insert auth header before the \r\n\r\n terminator.
                # auth_line already ends with \r\n, and data[header_end:]
                # starts with \r\n\r\n, so we must NOT add extra \r\n.
                request = (data[:header_end] + b"\r\n" +
                           auth_line.rstrip(b"\r\n") +
                           data[header_end:])
                upstream.sendall(request)

                # Read upstream's response to CONNECT (e.g. "HTTP/1.1 200")
                response = b""
                while b"\r\n\r\n" not in response:
                    chunk = upstream.recv(4096)
                    if not chunk:
                        return
                    response += chunk

                client.sendall(response)

                if b" 200 " not in response.split(b"\r\n")[0]:
                    return

                # Tunnel established — relay raw bytes with blocking threads
                self._relay(client, upstream)
            else:
                # Plain HTTP: inject auth header, forward, relay
                header_end = data.find(b"\r\n\r\n")
                if header_end != -1:
                    data = (data[:header_end] + b"\r\n" +
                            auth_line.rstrip(b"\r\n") +
                            data[header_end:])
                upstream.sendall(data)
                self._relay(client, upstream)
        except Exception:
            pass
        finally:
            for s in (client, upstream):
                if s is None:
                    continue
                try:
                    s.close()
                except OSError:
                    pass

    @staticmethod
    def _pipe(src: socket.socket, dst: socket.socket, done: threading.Event):
        """One-way blocking byte pipe."""
        try:
            while not done.is_set():
                chunk = src.recv(65536)
                if not chunk:
                    break
                dst.sendall(chunk)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            done.set()

    @classmethod
    def _relay(cls, a: socket.socket, b: socket.socket):
        """Bidirectional byte relay using two blocking threads."""
        done = threading.Event()
        t1 = threading.Thread(target=cls._pipe, args=(a, b, done), daemon=True)
        t2 = threading.Thread(target=cls._pipe, args=(b, a, done), daemon=True)
        t1.start()
        t2.start()
        done.wait(timeout=300)
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass
        t1.join(timeout=1)
        t2.join(timeout=1)


def create_browser(
    proxy_url: str | None = None,
    headless: bool = True,
    enable_cdp: bool = True,
    proxy_session: str | None = None,
    use_undetected: bool = False,
    block_images: bool = True,
) -> webdriver.Chrome:
    """
    Create and return a configured Chrome WebDriver instance.

    Each call creates a completely fresh browser session with no
    cached data, cookies, or browsing history. This is critical
    for the audit methodology - every hourly snapshot must start
    from a clean slate before cookies are injected.

    Args:
        proxy_url: Full proxy URL (http://user:pass@host:port).
                   If None, reads from settings.
        headless: Run in headless mode (True for production,
                  False for debugging).
        enable_cdp: Enable Chrome DevTools Protocol for network
                    JSON interception.
        proxy_session: Evomi sticky session ID. When set, appends
                       _session-<ID> to the proxy password so all
                       requests use the same residential IP.
        block_images: Disable image loading to save proxy bandwidth.
                      Default True. Set False for Instagram (React SPA
                      needs images for grid tiles to mount).

    Returns:
        Configured Chrome WebDriver instance.
    """
    settings = load_settings()

    if proxy_url is None:
        proxy_url = settings.PROXY_URL

    options = Options()

    # ----- Page load strategy -----
    # Use "eager" so driver.get() returns at DOMContentLoaded instead of
    # waiting for all sub-resources (images, ads, trackers).  Each scraper
    # already uses explicit WebDriverWait for the elements it needs.
    options.page_load_strategy = "eager"

    # ----- Core mode flags -----
    if headless:
        options.add_argument("--headless=new")  # New headless mode (Chrome 112+)
    # NOTE: --incognito removed. Selenium creates a fresh temp profile per call,
    # providing the same isolation. Incognito can block the proxy auth extension.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # ----- Viewport and display -----
    options.add_argument(
        f"--window-size={settings.VIEWPORT_WIDTH},{settings.VIEWPORT_HEIGHT}"
    )

    # ----- Locale and language -----
    # Sets the browser's Accept-Language header and UI language
    options.add_argument(f"--lang={settings.LANGUAGE}")
    options.add_experimental_option("prefs", {
        "intl.accept_languages": f"{settings.LOCALE},{settings.LANGUAGE}",
        "profile.default_content_setting_values.notifications": 2,  # Block popups
    })

    # ----- User agent -----
    options.add_argument(f"--user-agent={settings.USER_AGENT}")

    # ----- Proxy configuration (Evomi) -----
    # Uses a local TCP forwarder thread that injects Proxy-Authorization
    # headers, because Chromium 145 broke MV2 extension proxy auth APIs.
    proxy_forwarder = None
    if proxy_url and settings.PROXY_USER and settings.PROXY_PASS:
        proxy_pass = settings.PROXY_PASS
        if proxy_session:
            proxy_pass = f"{proxy_pass}_session-{proxy_session}"
        proxy_forwarder = _ProxyForwarder(
            upstream_host=settings.PROXY_HOST,
            upstream_port=int(settings.PROXY_PORT),
            username=settings.PROXY_USER,
            password=proxy_pass,
        )
        proxy_forwarder.start()
        logger.info(
            "Proxy forwarder started on 127.0.0.1:%d -> %s:%s",
            proxy_forwarder.port, settings.PROXY_HOST, settings.PROXY_PORT,
        )
        options.add_argument(f"--proxy-server=http://127.0.0.1:{proxy_forwarder.port}")
    elif proxy_url:
        # IP-whitelisted proxy: simple --proxy-server flag
        logger.info("Proxy configured (IP-whitelisted): %s:%s", settings.PROXY_HOST, settings.PROXY_PORT)
        options.add_argument(
            f"--proxy-server={settings.PROXY_HOST}:{settings.PROXY_PORT}"
        )

    # ----- CDP performance logging (JSON interception) -----
    # Required for driver.get_log("performance") to capture network events.
    # All three JSON parsers depend on this to intercept API responses.
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    # ----- Anti-detection measures -----
    # Disable automation indicators that platforms check
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")

    # WebRTC leak prevention: force WebRTC through proxy, prevent real IP exposure
    options.add_argument("--webrtc-ip-handling-policy=disable_non_proxied_udp")
    options.add_argument("--enforce-webrtc-ip-permission-check")

    # Disable image loading to reduce proxy bandwidth (~60-80% savings).
    # Visual features are extracted separately via thumbnail URLs in
    # extract_visual_features.py — they do NOT need browser-rendered images.
    # Instagram callers pass block_images=False because the React SPA
    # needs images for Explore grid tiles to mount (see findings.txt 2026-03-20).
    if block_images:
        options.add_argument("--blink-settings=imagesEnabled=false")

    # ----- Performance flags -----
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-popup-blocking")

    # ----- Create the driver -----
    if use_undetected:
        # undetected-chromedriver patches the ChromeDriver binary to remove
        # $cdc_ detection variables that PerimeterX (LinkedIn) checks.
        # UC manages its own ChromeDriver — don't pass Service().
        import undetected_chromedriver as uc
        # UC handles headless via constructor param, remove our --headless flag
        options_args = options.arguments
        for arg in list(options_args):
            if arg.startswith("--headless"):
                options_args.remove(arg)
        # UC handles anti-automation flags internally — remove ours to avoid conflict
        exp_opts = options.experimental_options
        exp_opts.pop("excludeSwitches", None)
        exp_opts.pop("useAutomationExtension", None)
        # Pin ChromeDriver version to match installed Chrome
        import subprocess as _sp
        chrome_ver = _sp.run(
            ["google-chrome", "--version"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split()[-1]  # e.g. "145.0.7632.116"
        version_main = int(chrome_ver.split(".")[0])
        driver = uc.Chrome(
            options=options, headless=headless, version_main=version_main,
        )
    else:
        # ----- Pre-flight: verify ChromeDriver can start -----
        _chromedriver = settings.CHROMEDRIVER_PATH
        if isinstance(_chromedriver, str) and _chromedriver:
            import subprocess as _sp
            try:
                _sp.run([_chromedriver, "--version"],
                        capture_output=True, timeout=5, check=True)
            except Exception as e:
                raise RuntimeError(
                    f"ChromeDriver pre-flight failed: {e}. "
                    f"If using snap chromium, ensure /run/user/{os.getuid()} exists "
                    f"(sudo loginctl enable-linger $USER)"
                ) from e
        service = Service(settings.CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)

    # Track proxy forwarder for cleanup (see cleanup_browser())
    driver._proxy_forwarder = proxy_forwarder

    # Set timeouts (explicit waits only — no implicitly_wait)
    driver.set_page_load_timeout(settings.PAGE_LOAD_TIMEOUT)

    # ----- Post-creation anti-detection -----
    # Override navigator properties, fingerprint vectors, and browser
    # objects to match a real Chrome session on Linux.
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                // 1. navigator.webdriver = undefined
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });

                // 2. navigator.plugins — headless has empty array, real Chrome has 5+
                Object.defineProperty(navigator, 'plugins', {
                    get: () => {
                        const plugins = [
                            { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer',
                              description: 'Portable Document Format', length: 1 },
                            { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                              description: '', length: 1 },
                            { name: 'Chromium PDF Viewer', filename: 'internal-pdf-viewer',
                              description: 'Portable Document Format', length: 1 },
                            { name: 'Native Client', filename: 'internal-nacl-plugin',
                              description: '', length: 2 },
                            { name: 'Widevine Content Decryption Module',
                              filename: 'libwidevinecdm.so',
                              description: 'Enables Widevine licenses for playback of HTML audio/video content.', length: 1 },
                        ];
                        plugins.refresh = () => {};
                        plugins.item = (i) => plugins[i] || null;
                        plugins.namedItem = (n) => plugins.find(p => p.name === n) || null;
                        return plugins;
                    }
                });

                // 3. navigator.mimeTypes — align with plugin array
                Object.defineProperty(navigator, 'mimeTypes', {
                    get: () => {
                        const mimes = [
                            { type: 'application/pdf', suffixes: 'pdf',
                              description: 'Portable Document Format' },
                            { type: 'application/x-nacl', suffixes: '',
                              description: 'Native Client Executable' },
                            { type: 'application/x-pnacl', suffixes: '',
                              description: 'Portable Native Client Executable' },
                        ];
                        mimes.item = (i) => mimes[i] || null;
                        mimes.namedItem = (n) => mimes.find(m => m.type === n) || null;
                        mimes.refresh = () => {};
                        return mimes;
                    }
                });

                // 4. navigator.languages — ensure proper array format
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['de-CH', 'de', 'en-US', 'en']
                });

                // 5. window.chrome — headless may lack this; add realistic runtime
                if (!window.chrome) {
                    window.chrome = {};
                }
                window.chrome.runtime = {
                    OnInstalledReason: {CHROME_UPDATE: 'chrome_update', INSTALL: 'install',
                                       SHARED_MODULE_UPDATE: 'shared_module_update', UPDATE: 'update'},
                    OnRestartRequiredReason: {APP_UPDATE: 'app_update', OS_UPDATE: 'os_update',
                                             PERIODIC: 'periodic'},
                    PlatformArch: {ARM: 'arm', ARM64: 'arm64', MIPS: 'mips', MIPS64: 'mips64',
                                   X86_32: 'x86-32', X86_64: 'x86-64'},
                    PlatformNaclArch: {ARM: 'arm', MIPS: 'mips', MIPS64: 'mips64',
                                       X86_32: 'x86-32', X86_64: 'x86-64'},
                    PlatformOs: {ANDROID: 'android', CROS: 'cros', LINUX: 'linux',
                                 MAC: 'mac', OPENBSD: 'openbsd', WIN: 'win'},
                    RequestUpdateCheckStatus: {NO_UPDATE: 'no_update', THROTTLED: 'throttled',
                                               UPDATE_AVAILABLE: 'update_available'},
                    connect: function() { return {onMessage: {addListener: function(){}},
                                                  postMessage: function(){},
                                                  disconnect: function(){}}; },
                    sendMessage: function() {},
                    id: undefined,
                };

                // 6. Permissions.query — return realistic defaults for fresh profile
                const origQuery = window.Permissions.prototype.query;
                window.Permissions.prototype.query = (parameters) => {
                    if (parameters.name === 'notifications') {
                        return Promise.resolve({ state: Notification.permission });
                    }
                    if (['geolocation', 'camera', 'microphone', 'midi',
                         'background-sync', 'ambient-light-sensor',
                         'accelerometer', 'gyroscope', 'magnetometer',
                         'clipboard-read', 'clipboard-write'].includes(parameters.name)) {
                        return Promise.resolve({ state: 'prompt' });
                    }
                    return origQuery(parameters);
                };

                // 7. Canvas fingerprint defense — add deterministic noise
                //    to canvas exports so fingerprint is unique but consistent
                //    within a session. Does not affect visual rendering.
                const _origToDataURL = HTMLCanvasElement.prototype.toDataURL;
                HTMLCanvasElement.prototype.toDataURL = function(type) {
                    const ctx = this.getContext && this.getContext('2d');
                    if (ctx && this.width > 0 && this.height > 0) {
                        try {
                            const img = ctx.getImageData(0, 0, Math.min(this.width, 16), 1);
                            for (let i = 0; i < img.data.length; i += 67) {
                                img.data[i] = img.data[i] ^ 1;
                            }
                            ctx.putImageData(img, 0, 0);
                        } catch(e) {}
                    }
                    return _origToDataURL.apply(this, arguments);
                };

                // 8. WebGL parameter spoofing — mask GPU/renderer info
                const _origGetParam = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(param) {
                    if (param === 0x9245) return 'Google Inc. (Intel)';
                    if (param === 0x9246) return 'ANGLE (Intel, Mesa Intel(R) UHD Graphics 630 (CFL GT2), OpenGL ES 3.2)';
                    return _origGetParam.apply(this, arguments);
                };
                if (typeof WebGL2RenderingContext !== 'undefined') {
                    const _origGetParam2 = WebGL2RenderingContext.prototype.getParameter;
                    WebGL2RenderingContext.prototype.getParameter = function(param) {
                        if (param === 0x9245) return 'Google Inc. (Intel)';
                        if (param === 0x9246) return 'ANGLE (Intel, Mesa Intel(R) UHD Graphics 630 (CFL GT2), OpenGL ES 3.2)';
                        return _origGetParam2.apply(this, arguments);
                    };
                }

                // 9. Media device enumeration — headless returns empty array
                if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                    const _origEnum = navigator.mediaDevices.enumerateDevices.bind(navigator.mediaDevices);
                    navigator.mediaDevices.enumerateDevices = function() {
                        return _origEnum().then(devices => {
                            if (devices.length === 0) {
                                return [
                                    {deviceId: '', groupId: 'default', kind: 'audioinput',  label: ''},
                                    {deviceId: '', groupId: 'default', kind: 'videoinput',  label: ''},
                                    {deviceId: '', groupId: 'default', kind: 'audiooutput', label: ''},
                                ];
                            }
                            return devices;
                        });
                    };
                }

                // 10. Screen properties — match viewport dimensions
                Object.defineProperty(screen, 'width',       { get: () => %d });
                Object.defineProperty(screen, 'height',      { get: () => %d });
                Object.defineProperty(screen, 'availWidth',  { get: () => %d });
                Object.defineProperty(screen, 'availHeight', { get: () => %d });
                Object.defineProperty(screen, 'colorDepth',  { get: () => 24 });
                Object.defineProperty(screen, 'pixelDepth',  { get: () => 24 });

                // 11. window.outerWidth/outerHeight — may be 0 in headless
                if (window.outerWidth === 0) {
                    Object.defineProperty(window, 'outerWidth',  { get: () => %d });
                    Object.defineProperty(window, 'outerHeight', { get: () => %d });
                }
            """ % (settings.VIEWPORT_WIDTH, settings.VIEWPORT_HEIGHT,
                   settings.VIEWPORT_WIDTH, settings.VIEWPORT_HEIGHT,
                   settings.VIEWPORT_WIDTH, settings.VIEWPORT_HEIGHT)
        },
    )

    # Set timezone via CDP
    driver.execute_cdp_cmd(
        "Emulation.setTimezoneOverride",
        {"timezoneId": settings.TIMEZONE},
    )

    # Set locale via CDP
    driver.execute_cdp_cmd(
        "Emulation.setLocaleOverride",
        {"locale": settings.LOCALE},
    )

    # ----- Enable CDP Network domain for JSON interception -----
    if enable_cdp:
        setup_cdp_network_interception(driver, block_images=block_images)

    logger.info(
        "Browser created: headless=%s, viewport=%dx%d, locale=%s, cdp=%s",
        headless,
        settings.VIEWPORT_WIDTH,
        settings.VIEWPORT_HEIGHT,
        settings.LOCALE,
        enable_cdp,
    )

    return driver


def setup_cdp_network_interception(driver: webdriver.Chrome, block_images: bool = True) -> None:
    """
    Enable CDP network domain for intercepting JSON API responses.

    Once enabled, network responses can be captured via
    driver.execute_cdp_cmd("Network.getResponseBody", ...).

    Args:
        driver: Active Chrome WebDriver with CDP enabled.
    """
    driver.execute_cdp_cmd("Network.enable", {})

    # Block heavy resource types to reduce proxy bandwidth.
    # When block_images=True, images are also blocked via blink-settings.
    # This additionally blocks video, fonts, and media streams at the
    # network level.  Does NOT affect JSON/XHR interception or DOM parsing.
    blocked_urls = [
        "*.mp4", "*.webm", "*.m3u8", "*.ts",   # video segments
        "*.woff", "*.woff2", "*.ttf", "*.otf",  # fonts
        "*.mp3", "*.aac", "*.ogg",               # audio
    ]
    if block_images:
        blocked_urls.extend(["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.svg"])
    driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": blocked_urls})
    logger.debug("CDP: blocked resource URLs (block_images=%s)", block_images)

    logger.debug("CDP Network domain enabled for JSON interception")


def cleanup_browser(driver: webdriver.Chrome) -> None:
    """
    Quit the browser and stop the proxy forwarder thread.

    Use this instead of bare driver.quit() to ensure the local proxy
    forwarder is stopped after each scrape run.

    Args:
        driver: Chrome WebDriver instance from create_browser().
    """
    forwarder = getattr(driver, "_proxy_forwarder", None)

    try:
        driver.quit()
    except Exception:
        pass

    if forwarder is not None:
        forwarder.stop()
        logger.debug("Proxy forwarder stopped")
