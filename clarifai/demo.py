from client.api import ApiClient
from client.user_api import UserApi


user_client = UserApi("sai_nivedh")
app = user_client.get_app("snowflake_sample")

print(app.list_apps())