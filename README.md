# Problem
This project provides an end-to-end solution to ensure people are verifying their identities online of their own free will, not under pressure from others. We're developing tools to detect when more than one person might be involved in the verification process, keeping online identity checks safe and truly voluntary.

![forced_verification](https://github.com/PBozmarov/multiperson-detect/assets/77898273/2394b98e-1dc6-46ac-ab33-d3b3e9f81b89){ width=300px height=50px }

# Solution
End-to-end model that returns 1 if the video contains more than one person at any given moment, 0 otherwise.
We use two approaches:
1) We fine-tune our custom private dataset Yolov8 Nano model, pre-trained on the COCO dataset. The model takes as input a frame/image and outputs bounding boxes for a **real person** a person appearing on the image, and a **fake person** - an image of a person, such as an ID of a person. Then we wrap the model into a framework that tracks the number of real people per frame.
2) We use the Yolov8 Nano model, and we only detect the class **person**, after detection, we extract the ROI(region of interest) within the predicted bounding boxes and use the Silent-Face-Anti-Spoofing detector (referenced below) to detect whether a person is a real or fake. 

# Set Up
To set up your environment, follow these steps:

1. Open your terminal.
2. Ensure you are in the directory containing the `test_env.yaml` file.
3. Run the following command to create a new conda environment named `<myenv>` using the configuration in `test_env.yaml`:

   **`conda env create -f test_env.yaml --name <myenv>`**

   Replace `<myenv>` with your desired environment name.

After executing this command, your new environment should be set up and ready to use.


# Usage

# References
* https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md
