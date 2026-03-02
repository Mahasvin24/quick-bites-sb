import os
import time

from carrillo_counter import CarrilloCounter


PREVIOUS_IMAGE_PATH = "images/carrillo_previous.jpg"
CURRENT_IMAGE_PATH = "images/carrillo_current.jpg"
CLASSIFY_INTERVAL_SECONDS = 5


def _read_image_bytes(path):
    with open(path, "rb") as image_file:
        return image_file.read()


def main():
    os.makedirs("images", exist_ok=True)
    counter = CarrilloCounter()
    counter.warmup()

    try:
        while True:
            if not os.path.exists(PREVIOUS_IMAGE_PATH) or not os.path.exists(CURRENT_IMAGE_PATH):
                print(
                    "Waiting for carrillo images... "
                    "(run update-images.py to fetch current/previous frames)"
                )
                time.sleep(CLASSIFY_INTERVAL_SECONDS)
                continue

            previous_image_bytes = _read_image_bytes(PREVIOUS_IMAGE_PATH)
            current_image_bytes = _read_image_bytes(CURRENT_IMAGE_PATH)

            result = counter.process(previous_image_bytes, current_image_bytes)
            if result is None:
                print("carrillo decode error")
            else:
                print(
                    (
                        f"carrillo occupancy={result['occupancy']} "
                        f"(in:+{result['cycle_in']}, out:-{result['cycle_out']})"
                    )
                )

            time.sleep(CLASSIFY_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nCounter stopped.")


if __name__ == "__main__":
    main()
