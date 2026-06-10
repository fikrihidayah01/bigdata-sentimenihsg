# login_session.py — Jalankan SEKALI untuk simpan session login
import asyncio
from playwright.async_api import async_playwright

async def save_login():
    async with async_playwright() as p:
        # Buka browser dengan folder profile permanen
        context = await p.chromium.launch_persistent_context(
            user_data_dir="./browser_profile",   # session disimpan di sini
            headless=False,
            args=[
                "--lang=en-US",
                "--disable-blink-features=AutomationControlled",  # sembunyikan flag automation
            ],
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )

        page = await context.new_page()

        # Buka threads langsung
        await page.goto("https://www.threads.net/login", wait_until="domcontentloaded")

        print("\n" + "="*55)
        print("  LOGIN MANUAL DI BROWSER YANG TERBUKA")
        print("  - Selesaikan captcha jika ada")
        print("  - Tunggu sampai halaman HOME muncul")
        print("  - Kembali ke terminal ini")
        print("="*55)
        input("\n  >> Tekan Enter setelah halaman HOME Threads muncul: ")

        # Simpan storage state (cookies, localStorage, dll)
        await context.storage_state(path="./auth_state.json")
        print("\n[✓] Session berhasil disimpan ke auth_state.json")
        print("[✓] Kamu tidak perlu login lagi saat scraping!")

        await context.close()

asyncio.run(save_login())