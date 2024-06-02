import os

def read_txt_files(directory_path, encoding='utf-8'):
    # Initialize an empty string to store the content of all text files
    all_content = ""

    # List all files in the directory
    files = os.listdir(directory_path)

    # Iterate through the files and read the content of text files
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(directory_path, file)
            with open(file_path, 'r', encoding=encoding) as txt_file:
                # Read the content of the text file and append it to the all_content string
                file_content = txt_file.read()
                all_content += file_content

    # Remove "\t", "\r", and "\n" characters from the merged content
    cleaned_content = all_content.replace("\t", "").replace("\r", "").replace("\n", "")

    return cleaned_content

def read_file_content(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        content = file.read()
        return content