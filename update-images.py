import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

consumer_key = os.getenv("CONSUMER_KEY")

BASE_URL = "https://api.ucsb.edu/dining/cams/v2/still"
DINING_HALLS = ["carrillo", "de-la-guerra", "ortega", "portola"]

frames = {
    hall: {"current": None, "previous": None} for hall in DINING_HALLS
}

try:
    while True:
        # Updating images for each hall
        for hall in DINING_HALLS:
            url = f"{BASE_URL}/{hall}?ucsb-api-key={consumer_key}"
            res = requests.get(url)

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
        time.sleep(5) # DO NOT REMOVE THIS LINE OF CODE

except KeyboardInterrupt:
    print("\nProgram stopped.")