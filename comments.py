from googleapiclient.discovery import build

api_key = 'AIzaSyDQPOd5bP7jbabOd1cNZcuM4ttRy8Tw1Yo'

def get_video_comments(video_id):
	
	comments = []

	# creating youtube resource object
	youtube = build('youtube', 'v3',
					developerKey=api_key)

	# retrieve youtube video results
	video_response=youtube.commentThreads().list(
	part='snippet',
	videoId=video_id
	).execute()

	# iterate video response
	while video_response:
	
		# extracting required info
		# from each result object 
		for item in video_response['items']:
		
			# Extracting comments
			comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
			comments.append(comment)
			print(comment,end = '\n\n')

		# Again repeat
		if 'nextPageToken' in video_response:
			video_response = youtube.commentThreads().list(
					part = 'snippet',
					videoId = video_id,
					pageToken = video_response['nextPageToken']
				).execute()
		else:
			break
	
	return comments

