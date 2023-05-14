from kandinsky2 import get_kandinsky2
import cv2 as cv
import socket
import socks
import os

os.environ['HTTP_PROXY']="127.0.0.1:60119"
os.environ['HTTPS_PROXY']="127.0.0.1:60119"
#socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 60118)
#socket.socket = socks.socksocket

model = get_kandinsky2('cuda', task_type='text2img')
images = model.generate_text2img('A teddy bear на красной площади', batch_size=4, h=512, w=512, num_steps=75, denoised_type='dynamic_threshold', dynamic_threshold_v=99.5, sampler='ddim_sampler', ddim_eta=0.05, guidance_scale=10)
cv.imwrite('./test.img', images)
