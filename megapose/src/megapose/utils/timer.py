"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



# Standard Library
import datetime


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = datetime.timedelta()
        self.is_running = False

    def reset(self):
        self.start_time = None
        self.elapsed = 0.0
        self.is_running = False

    def start(self):
        self.elapsed = datetime.timedelta()
        self.is_running = True
        self.start_time = datetime.datetime.now()
        return self

    def pause(self):
        if self.is_running:
            self.elapsed += datetime.datetime.now() - self.start_time
            self.is_running = False

    def resume(self):
        if not self.is_running:
            self.start_time = datetime.datetime.now()
            self.is_running = True

    def stop(self):
        self.pause()
        elapsed = self.elapsed
        self.reset()
        return elapsed
