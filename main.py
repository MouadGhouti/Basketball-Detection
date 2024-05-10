from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker

def main():
    # Read Video
    input_video_path = "input/IMG_8468.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/best.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )   
    
    Ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     ) 
     
    #Draw Output
    print((Ball_detections[0])[1]) 
    ##Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)

    output_video_frames = ball_tracker.draw_bboxes(video_frames,Ball_detections)

    save_video(output_video_frames, 'output/outputvideo3.avi')

if __name__ == "__main__":
    main()