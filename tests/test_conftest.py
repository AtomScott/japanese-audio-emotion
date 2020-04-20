def test_image_data(single_image):
    assert single_image.shape == (366, 630, 3)


def test_video_data(video_data):
    n_frames = len([0 for _ in video_data])
    assert n_frames == 899

