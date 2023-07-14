from clarifai.client.api import ApiClient

client = ApiClient()

# List all users
print(client.list_users())

# List all apps for a user
print(client.user('user_id').list_apps())

# Get an app
app = client.user('user_id').app('app_id')