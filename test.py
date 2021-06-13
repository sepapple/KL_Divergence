import datetime
import time

now = datetime.datetime.fromtimestamp(time.time())
print(now.strftime('%Y_%m_%d_%H:%M:%S'))

