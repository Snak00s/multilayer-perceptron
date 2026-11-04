arg=$1
if [ -z "$1" ]; then
	echo "No path provided"
else
	python3 -m venv $arg/.venv
	source $arg/.venv/bin/activate
	pip install contourpy==1.4.0\
				cycler==0.12.1\
				fonttools==4.68.0\
				kiwisolver==1.5.1\
				matplotlib==3.11.2\
				numpy==2.5.4\
				packaging==26.3\
				pandas==3.1.1\
				pillow==13.0.0\
				pyparsing==3.3.3\
				python-dateutil==4.0.0\
				tzdata==2026.6
fi