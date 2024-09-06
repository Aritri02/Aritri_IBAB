#!/bin/bash
file="users.csv"
if [ ! -f "$file" ]
 then
    echo "File $file not found!"
    exit 1
fi
tail -n +2 "$file" | while IFS=',' read -r userid username_userdept
do
IFS=':' read -r username userdept <<< "$username_userdept"
 echo "UserID: $userid"
    echo "Username: $username"
    echo "Department: $userdept"
    echo 
done