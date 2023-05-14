from kandinsky2 import get_kandinsky2
import cv2 as cv
import socket
import socks
import os

os.environ['HTTP_PROXY']="127.0.0.1:60119"
os.environ['HTTPS_PROXY']="127.0.0.1:60119"
#socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 60118)
#socket.socket = socks.socksocket

model = get_kandinsky2('cpu',#'cuda',
                       task_type='text2img',
                       cache_dir='I:\\ai\\cv\\models\\kandinsky2',
                       proxies={
                           'http': '127.0.0.1:60119',
                           'https': '127.0.0.1:60119',
                        },
                        local_files_only=True,
                    )
images = model.generate_text2img('A teddy bear на красной площади',
                                 num_steps=100,
                                 batch_size=1, 
                                 guidance_scale=4,
                                 h=768, w=768,
                                 sampler='p_sampler', 
                                 prior_cf_scale=4,
                                 prior_steps="5",
                                )
cv.imwrite('./test.img', images)
