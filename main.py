from langchain_text_splitters import RecursiveCharacterTextSplitter


with open("resume.txt", "r") as file:
    resume_data = file.read()


chunked = RecursiveCharacterTextSplitter(
    separators = ["\n\n", " ", "", "\n"],
    chunk_size = 100,
    chunk_overlap  = 20
)


a =  chunked.split_text(resume_data)


for i ,line in enumerate(a):
    print(i , line)