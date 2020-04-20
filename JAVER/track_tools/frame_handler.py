class FrameHandler:
    def __init__(self, step, batch_size, video_path):
        self._idx = 0
        self._step = step
        self._batch_size = batch_size

        # Can not store all frames because of memory limits
        video = mmcv.VideoReader(video_path)
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

        # what is i???
        # count = 0
        # for i in reader:
        #     frames.append(reader.get_next_data())
        #     # ! Temporary break
        #     if count == 10 ** 3:
        #         break
        #     else:
        #         count += 1
        self._frames = frames

    def __len__(self):
        return len(self.frames)

    @property
    def frames(self):
        return self._frames

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, x):
        if self.idx + x <= self.__len__():
            logger.warning("Step size too big!!")
        self._step = x

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, x):
        if x <= self.__len__():
            self._idx = x
        else:
            raise ValueError(f"idx > self.__len__(): {x}>{self.__len__()}")

    def next(self):
        if not self.has_next():
            return

        frames = self.frames
        n_frames = len(frames)
        step = self.step
        batch_size = self.batch_size
        start_idx = self.idx

        end_idx = start_idx + (step * batch_size)

        if end_idx >= n_frames:
            end_idx = n_frames - (n_frames % step)

        self.idx = end_idx
        return self.frames[start_idx:end_idx:step]

    def has_next(self):
        return self.idx + self.step <= self.__len__()
