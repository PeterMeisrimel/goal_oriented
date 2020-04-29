## cleanup
rm -r input_files/verification 
rm -r __pycache__
rm *.pyc
rm output.txt

python3 initialize.py > /dev/null # hide ouput
python3 main.py     >output.txt
python3 output_comp.py
