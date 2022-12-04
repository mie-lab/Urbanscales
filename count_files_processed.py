import os
import time


starttime = time.time()

_, _, files = next(os.walk("temp"))
old_file_count = len(files)
for i in range(1000):
    time.sleep(2)

    _, _, files = next(os.walk("temp"))
    new_file_count = len(files)

    print((new_file_count - old_file_count), " processed per second")

    old_file_count = new_file_count
