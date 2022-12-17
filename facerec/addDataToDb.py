import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("keyfiredb.json")
firebase_admin.initialize_app(cred,{
 	'databaseURL':"https://faceattendancerealtime-b01d1-default-rtdb.firebaseio.com/"
 	})

ref = db.reference('Students')


data = {
    "321654":
        {
            "name": "Murtaza Hassan",
            "last_attendance_time": "2022-12-11 00:54:34",
            "total_attendance": 7
        },
    "852741":
        {
            "name": "Emly Blunt",
            "last_attendance_time": "2022-12-11 00:54:34",
            "total_attendance": 7
        },
    "963852":
        {
            "name": "Elon Musk",
            "last_attendance_time": "2022-12-11 00:54:34",
            "total_attendance": 7
        },
	"111":
		{
			"name": "Filonov Roman",
            "last_attendance_time": "2022-12-11 00:54:34",
            "total_attendance": 1
		}
}

for key, value in data.items():
	ref.child(key).set(value)