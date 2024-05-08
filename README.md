# Problem
This project provides an end-to-end solution to ensure people are verifying their identities online of their own free will, not under pressure from others. We're developing tools to detect when more than one person might be involved in the verification process, keeping online identity checks safe and truly voluntary.

<img src="https://github.com/PBozmarov/multiperson-detect/assets/77898273/2394b98e-1dc6-46ac-ab33-d3b3e9f81b89" width="500" height="500" alt="forced_verification">


When verifying people usually hold IDs, however traditional pre-trained models such as yolov8n.pt don't differentiate between an ID of a person or a real person, thus we needed to make sure our framework recognises both real and fake people.

<img src="https://github.com/PBozmarov/multiperson-detect/assets/77898273/599a548e-5cf6-4539-a338-85ecf6ca0f22" width="400" height="400" alt="real_fake">



# Solution
End-to-end model that returns 1 if the video contains more than one person at any given moment, 0 otherwise.
We use two approaches:
1) We fine-tune our custom private dataset Yolov8 Nano model, pre-trained on the COCO dataset. The model takes as input a frame/image and outputs bounding boxes for a **real person** a person appearing on the image, and a **fake person** - an image of a person, such as an ID of a person. Then we wrap the model into a framework that tracks the number of real people per frame.
2) We use the Yolov8 Nano model, and we only detect the class **person**, after detection, we extract the ROI(region of interest) within the predicted bounding boxes and use the Silent-Face-Anti-Spoofing detector (referenced below) to detect whether a person is a real or fake.

# Repository Structure
```
.
├── anti_spoof # the referenced at the bottom anti-spoofing detector repository
├── data # the structure of the private data folder/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
│   ├── yolov8_n_pt_epochs_100_sz_640/ 
│   │   ├── weights/
│   │   │   ├── best.pt # the weights of our trained model
│   │   │   └── last.pt
│   │   └── ..
│   └── yolov8n.pt 
├── results/
│   ├── labels.txt # the labels of our private data
│   ├── test_fine_tuned_predictions_skip_frames_1.txt # preds for our fine-tuned model with skip_frames = 1
│   ├── test_fine_tuned_predictions_skip_frames_2.txt # preds for our fine-tuned model with skip_frames = 2
│   ├── test_pre_trained_predictions_skip_frames_1.txt  # preds for our pre-trained model with skip_frames = 1
│   └── test_pre_trained_predictions_skip_frames_2.txt # preds for our pre-trained model with skip_frames = 2
└── main.py # the main script to be used for prediction
```

# Installation
Clone and enter the repository:

      git clone https://github.com/PBozmarov/multiperson-detect.git
      cd multiperson-detect

Set up the environment:

1. Open your terminal.
2. Ensure you are in the directory containing the test_env.yaml file.
3. Run the following command to create a new conda environment named <myenv> using the configuration in test_env.yaml:

       conda env create -f test_env.yaml --name <myenv>

   Replace <myenv> with your desired environment name.

After executing this command, your new environment should be set up and ready to use.

# Usage
Note! The first time you run the script it can take 2-3 mins for the _ultralytics_ module to initialise. 

**Label a single video:**

    python main.py --mode single --video_path your_video_path

**Label videos from a folder:**

    python main.py --mode folder --video_path your_folder_path

| Argument | Values | Description | Default |
|---|---|---|---|
| mode | single, folder | single - video, folder - folder of videos. | single |
| model_type | tuned, pre_trained | tuned - yolov8n.pt tuned on custom dataset, pre_trained - yolov8n pre-trained on the COCO dataset | tuned |
| video_path | absolute/relative path to the video or the folder |  | None |
| skip_frames | positive int: 1, 2, 3, 4,..  | defines how many frames we will skip when processing the video, 1 means we skip every other frame, 2 means we skip every 2 frames, and so on. | 1 |
| verbose | bool | shows additional logs | False |
| show_video | bool | shows the video and the predicted bounding boxes | False |


# References
* https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md
