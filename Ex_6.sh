#!/bin/bash
read -p "Enter the name of the directory" dir
cd $dir
for dir in *
do
	if [ -d $dir ]
	then
		echo "$dir is a sub_directory" >> sub_dir.txt
	else
		echo "$dir is not a directory"
	fi
	
done
ls -lh | awk '{print $5 , $9}' >> sub_dir.txt