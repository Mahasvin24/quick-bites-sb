import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

consumer_key = os.getenv("CONSUMER_KEY")
if not consumer_key:
    raise RuntimeError("Missing CONSUMER_KEY in environment.")

BASE_URL = "https://api.ucsb.edu/dining/cams/v2/still"
DINING_HALLS = ["carrillo", "ortega", "portola"]
FETCH_INTERVAL_SECONDS = 5

frames = {
    hall: {"current": None, "previous": None} for hall in DINING_HALLS
}

# Quick directory check
os.makedirs("images", exist_ok=True)


try:
    while True:
        # Updating images for each hall
        for hall in DINING_HALLS:
            url = f"{BASE_URL}/{hall}?ucsb-api-key={consumer_key}"
            try:
                res = requests.get(url, timeout=10)
            except requests.RequestException as exc:
                print(f"{hall} request error: {exc}")
                continue

            if res.status_code == 200:
                frames[hall]["previous"] = frames[hall]["current"]
                frames[hall]["current"] = res.content

                print(f"{hall} updated")

                # Optional: save to disk
                if frames[hall]["previous"]:
                    with open(f"images/{hall}_previous.jpg", "wb") as f:
                        f.write(frames[hall]["previous"])

                with open(f"images/{hall}_current.jpg", "wb") as f:
                    f.write(frames[hall]["current"])

            else:
                print(f"{hall} error: {res.status_code}")

        print()
        time.sleep(FETCH_INTERVAL_SECONDS) # DO NOT REMOVE THIS LINE OF CODE

except KeyboardInterrupt:
    print("\nProgram stopped.")