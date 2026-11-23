if ! command -v python3 >/dev/null 2>&1;
then
    echo "Python3 is not installed. Please install Python3 to proceed."
    exit 1
fi

python3 -m venv venv

source venv/bin/activate

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please ensure the file exists in the current directory."
    deactivate
    exit 1
fi

echo "Virtual environment setup complete and dependencies installed."