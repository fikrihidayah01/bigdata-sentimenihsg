# scraper.py — Menggunakan Chrome asli via CDP
import asyncio
import json
import re
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright
from config import SEARCH_KEYWORD, DATE_START, DATE_END, MAX_POSTS


async def wait_for_stable(page, timeout=10000):
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
    except:
        await page.wait_for_timeout(3000)


def clean_text(raw: str) -> str:
    """Bersihkan noise dari teks post."""
    noise_keywords = [
        "translate", "terjemahkan",
        "see more", "see less",
        "lihat selengkapnya", "lihat lebih",
        "balas", "reply", "follow", "following",
    ]
    lines = raw.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip angka murni (likes, comments)
        if re.fullmatch(r'[\d\s,.]+', line):
            continue
        # Skip tanggal relatif: "4d", "2h", "1w", "3m"
        if re.fullmatch(r'\d+[dhwm]', line.lower()):
            continue
        # Skip tanggal absolut: "04/25/26"
        if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2,4}', line):
            continue
        # Skip indikator slide: "1/7", "2/3"
        if re.fullmatch(r'\d+\s*/\s*\d+', line):
            continue
        # Skip noise UI
        if any(line.lower().startswith(kw) or line.lower() == kw
               for kw in noise_keywords):
            continue
        # Skip teks sangat pendek tanpa huruf
        if len(line) < 3 and not any(c.isalpha() for c in line):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


async def extract_text(post) -> str:
    """Ambil teks post dengan selector yang terbukti bekerja,
    lalu bersihkan noise-nya."""
    text = ""
    try:
        text_els = await post.query_selector_all('span[dir="auto"]')
        candidates = []
        for el in text_els:
            t = (await el.inner_text()).strip()
            style = await el.get_attribute("style") or ""
            if len(t) > 15:
                if "clamp" in style or "base-line" in style:
                    candidates.insert(0, t)   # konten utama duluan
                else:
                    candidates.append(t)
        if candidates:
            # Gabungkan semua kandidat (bukan cuma [0]) lalu bersihkan
            raw = "\n".join(candidates)
            text = clean_text(raw)
    except:
        pass

    # Fallback: inner_text seluruh post
    if not text:
        try:
            raw = await post.inner_text()
            text = clean_text(raw)
            # Kalau masih ada, ambil baris terpanjang sebagai konten utama
            if text:
                lines = [l for l in text.splitlines() if len(l) > 20]
                text = max(lines, key=len) if lines else text
        except:
            pass

    return text


async def scrape_threads():
    results = []
    keyword_encoded = SEARCH_KEYWORD.replace(" ", "%20")
    search_url = f"https://www.threads.com/search?q={keyword_encoded}&serp_type=default"

    async with async_playwright() as p:

        # ── Sambungkan ke Chrome yang sudah berjalan ─────────────────
        try:
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")
            print("[✓] Berhasil terhubung ke Chrome!")
        except Exception as e:
            print("[ERROR] Tidak bisa terhubung ke Chrome.")
            print("        Pastikan kamu sudah menjalankan start_chrome.bat")
            print(f"        Detail: {e}")
            return []

        context = browser.contexts[0]

        # Cari tab Threads yang sudah terbuka (support .net dan .com)
        page = None
        for p_tab in context.pages:
            if "threads.net" in p_tab.url or "threads.com" in p_tab.url:
                page = p_tab
                print(f"[INFO] Menggunakan tab yang sudah terbuka: {p_tab.url}")
                break

        if page is None:
            page = await context.new_page()
            print("[INFO] Membuka tab baru...")

        # Navigasi ke halaman search
        print(f"[INFO] Membuka: {search_url}")
        await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
        await wait_for_stable(page)
        await page.wait_for_timeout(3000)

        print(f"[INFO] URL setelah navigasi: {page.url}")

        # Cek session valid
        if "login" in page.url.lower() or "recaptcha" in page.url.lower():
            print("[ERROR] Belum login atau kena captcha.")
            print("        Login dulu di Chrome yang terbuka, lalu jalankan ulang script.")
            return []

        print(f"[✓] Session valid!")
        print(f"[INFO] Mulai scraping: '{SEARCH_KEYWORD}'\n")

        last_count = 0
        stall_count = 0

        while len(results) < MAX_POSTS:
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
            except:
                pass

            try:
                posts = await page.query_selector_all('article, div[data-pressable-container]')
            except Exception as e:
                print(f"[WARNING] Query gagal, retry... ({e})")
                await page.wait_for_timeout(3000)
                continue

            for post in posts:
                if len(results) >= MAX_POSTS:
                    break
                try:
                    # ── Ambil username dulu ──────────────────────────
                    user_el = await post.query_selector("a[href^='/@']")
                    username = await user_el.get_attribute("href") if user_el else None
                    if username:
                        username = username.replace("/@", "")

                    # ── Ambil URL post ───────────────────────────────
                    link_el = await post.query_selector("a[href*='/post/']")
                    post_url = await link_el.get_attribute("href") if link_el else None
                    if post_url and not post_url.startswith("http"):
                        post_url = "https://www.threads.com" + post_url

                    # Skip duplikat
                    if post_url and any(r["url"] == post_url for r in results):
                        continue

                    # ── Ambil teks (pakai fungsi extract_text) ───────
                    text = await extract_text(post)

                    # Setelah dapat teks, hapus username dari teks jika muncul
                    if username and text:
                        text = re.sub(
                            r'^' + re.escape(username) + r'\s*\n?',
                            '', text, flags=re.IGNORECASE
                        ).strip()

                    if not text:
                        continue

                    # ── Ambil timestamp ──────────────────────────────
                    time_el = await post.query_selector("time")
                    timestamp = await time_el.get_attribute("datetime") if time_el else None

                    results.append({
                        "username":   username or "unknown",
                        "text":       text,
                        "timestamp":  timestamp,
                        "url":        post_url or "",
                        "keyword":    SEARCH_KEYWORD,
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"  [+] #{len(results):03d} @{username}: {text[:70]}...")

                except Exception:
                    continue

            # Scroll ke bawah
            try:
                await page.evaluate("window.scrollBy(0, 1500)")
            except:
                pass
            await page.wait_for_timeout(2500)

            # Deteksi stall
            if len(results) == last_count:
                stall_count += 1
                print(f"  [~] Tidak ada post baru ({stall_count}/5)...")
                if stall_count >= 5:
                    print("[INFO] Scraping selesai.")
                    break
            else:
                stall_count = 0
            last_count = len(results)

        print("\n[INFO] Chrome dibiarkan tetap terbuka.")

    # ── Filter berdasarkan range tanggal ────────────────────────────
    filtered = []
    for r in results:
        if r["timestamp"]:
            try:
                post_date = datetime.fromisoformat(
                    r["timestamp"].replace("Z", "+00:00")
                ).date()
                start = datetime.strptime(DATE_START, "%Y-%m-%d").date()
                end   = datetime.strptime(DATE_END,   "%Y-%m-%d").date()
                if start <= post_date <= end:
                    filtered.append(r)
            except:
                filtered.append(r)
        else:
            filtered.append(r)

    print(f"\n[INFO] Total terkumpul : {len(results)}")
    print(f"[INFO] Setelah filter  : {len(filtered)}")
    return filtered


def save_results(data):
    if not data:
        print("[WARNING] Tidak ada data.")
        return

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = SEARCH_KEYWORD.replace(" ", "_")

    df = pd.DataFrame(data)
    csv_path = f"output/threads_{slug}_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[✓] CSV  → {csv_path}")

    json_path = f"output/threads_{slug}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] JSON → {json_path}")


if __name__ == "__main__":
    data = asyncio.run(scrape_threads())
    save_results(data)