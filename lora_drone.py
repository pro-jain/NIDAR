import sys
import time
import threading

from SX127x.LoRa import *
from SX127x.board_config import BOARD


# =========================================================
# ---------------- NODE CONFIGURATION ---------------------
# =========================================================

# Node IDs
DRONE1_ID = 1
GROUND_ID = 10
DRONE2_ID = 2

# ---- CHANGE THESE PER DEVICE ----
MY_ID = DRONE1_ID      # DRONE1_ID / GROUND_ID / DRONE2_ID
ROLE  = "DRONE"        # "DRONE" or "GROUND"
# --------------------------------


# =========================================================
# ---------------- LoRa Dual Role -------------------------
# =========================================================
class LoRaDualRole(LoRa):

    def __init__(self, verbose=False):
        super(LoRaDualRole, self).__init__(verbose)

        self.set_mode(MODE.SLEEP)
        self.set_dio_mapping([0] * 6)   # polling mode
        self.lock = threading.Lock()

    # ---------------- RX HANDLER (Polling) ----------------
    def handle_rx(self):
        irq = self.get_irq_flags()

        if irq.get('rx_done'):
            self.clear_irq_flags(RxDone=1)
            payload = self.read_payload(nocheck=True)

            try:
                msg = bytes(payload).decode("utf-8", errors="ignore")

                # -------- PACKET FORMAT --------
                # SRC|DST|DATA
                src, dst, data = msg.split("|", 2)
                src = int(src)
                dst = int(dst)

                # -------- V-SHAPE ENFORCEMENT --------
                if dst != MY_ID:
                    return  # Ignore packets not meant for me

                print(f"\n RX from {src}: {data}")
                print("TX > ", end="", flush=True)

            except Exception as e:
                print(f"\n[RX ERROR] {e}")

            self.reset_ptr_rx()
            self.set_mode(MODE.RXCONT)

    # ---------------- TX ----------------
    def send_message(self, dst_id, message):
        with self.lock:
            self.set_mode(MODE.STDBY)
            self.set_pa_config(pa_select=1)

            packet = f"{MY_ID}|{dst_id}|{message}"
            payload = list(bytearray(packet, "utf-8"))
            self.write_payload(payload)

            print(f" TX → {dst_id}: {message}")
            self.set_mode(MODE.TX)

            while not self.get_irq_flags().get('tx_done'):
                time.sleep(0.01)

            self.clear_irq_flags(TxDone=1)

            self.reset_ptr_rx()
            self.set_mode(MODE.RXCONT)


# =========================================================
# ---------------- TX THREAD ------------------------------
# =========================================================
def tx_thread(lora):
    while True:
        msg = sys.stdin.readline().strip()
        if not msg:
            continue

        # -------- ROLE-BASED DESTINATION --------
        if ROLE == "GROUND":
            # Ground decides which drone
            dst = int(input("Send to Drone ID (1 or 2): ").strip())
            lora.send_message(dst, msg)
        else:
            # Drone always talks to ground
            lora.send_message(GROUND_ID, msg)


# =========================================================
# ---------------- MAIN -----------------------------------
# =========================================================
def main():

    BOARD.setup()
    lora = LoRaDualRole(verbose=False)

    # -------- RADIO CONFIG (SAME EVERYWHERE) --------
    lora.set_mode(MODE.STDBY)
    lora.set_freq(434.0)
    lora.set_pa_config(pa_select=1)
    lora.set_spreading_factor(7)
    lora.set_bw(BW.BW125)
    lora.set_coding_rate(CODING_RATE.CR4_5)
    lora.set_sync_word(0x12)

    lora.reset_ptr_rx()
    lora.set_mode(MODE.RXCONT)

    threading.Thread(
        target=tx_thread,
        args=(lora,),
        daemon=True
    ).start()

    print("\n LoRa V-SHAPE MODE ACTIVE")
    print(f"Node ID: {MY_ID} | Role: {ROLE}")
    print("Listening...\n")
    print("TX > ", end="", flush=True)

    try:
        while True:
            lora.handle_rx()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        lora.set_mode(MODE.SLEEP)
        BOARD.teardown()


if __name__ == "__main__":
    main()