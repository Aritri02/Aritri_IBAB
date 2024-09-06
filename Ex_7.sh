#!/bin/bash
read -p "Enter the name of the file" file
sort "$file" | uniq > out_file.txt
echo "Duplicate lines are removed"
echo "The contents of the new file is:"
cat out_file.txt