# :#/ GSLF
# Gioele SL Fierro
# 2021
#
# ML Spam Detector CLI Interface

import time
from utils import clear
from vars import DATASET_PATH, LABELS_PATH

from SpamDetector import SpamDetector

def launcher():

    detector = SpamDetector(DATASET_PATH, LABELS_PATH)

    while True:
        # Clear
        clear()

        # Print Menu
        print("######################")
        print("")
        print(":#/ ML Spam Detector")
        print("")
        print("1 > Train")
        print("2 > Classify")
        print("0 > EXIT")
        print("")
        print("######################")
        print("")

        choice = input("Your choice: ")

        # Launch Training
        if choice == "1":
            print("Training in progress . . .")
            training_result = detector.startTraining()
            print(training_result)
            input("Press any key to continue . . .")

        # Launch Classification
        elif choice == "2":
            mail_path = input("Mail file path: ")
            detector.loadWeights()
            
            prediction = detector.classify(mail_path)
            print(prediction)
            input("Press any key to continue . . .")

        # Exit
        elif choice == "0":
            break

        # Wrong choice
        else:
            clear()
            print("WRONG CHOICE!")
            time.sleep(1.5)
            

if __name__ == "__main__":
    launcher()