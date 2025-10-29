from log import logger
from timer import get_timers

self.timers = get_timers()
self.timers("read-data").start()
self.timers("read-data").stop()
timer_info = self.timers.log(self.timers.timers.keys(), reset=True)
logger.info(f"[Profile global_step: {self.state.global_step}] {timer_info} {paddle_timer_info}")