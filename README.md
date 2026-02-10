# captcha-solver
Solve Image based captchas for OwObot


`train.py` -> To actually train the model. No pre-processing of images was done (was lazy). Perhaps you should consider preprocessing, as that may fetch better results.
`test_onnx.p` -> After exporting to an ONNX model, we can test the captcha solver with this file. 
`examples` folder contains 5 example captchas from OwO-Bot

All captcha images in use in my dataset was fetched with the help of https://github.com/Tyrrrz/DiscordChatExporter
https://github.com/wkentaro/labelme was used to annotate captcha images.

Took around 2~ days to get this to 90~ % accuracy with only 500~ images used for training. With better control of images in use, perhaps it could be reduced to 300 as I managed to achieve80% accuracy with just 300.

Feel free to use this mode **in compliance with GNU GPL V3** licence :>
