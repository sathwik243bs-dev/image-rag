import easyocr
reader =easyocr.Reader(["en"])
result =reader.readtext("work-policy-D13896.png")
for detection in result:
    text=detection[1]
    print(text,end="")
