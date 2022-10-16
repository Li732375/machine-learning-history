##OCR 為光學文字識別的縮寫（Optical Character Recognition，OCR）
##常見的語言字串：英文 'eng'、簡體中文 'chi_sim'、繁體中文 'chi_tra'
##本程式僅辨識數字，不考慮文字
## https://my.oschina.net/u/2396236/blog/1621590
##將想要轉成文字的圖片檔皆可嘗試
##介面、測試皆完成

from PIL import Image
import pytesseract, os

while True:
    target = input('拖曳欲辨識的目標至此：')

    if len(target) == 0:
        break

    img = Image.open(target)

    #倘若要辨識語言，後面加上參數 lang = 'eng'
    text = pytesseract.image_to_string(img, config = '--psm 7 digits_onlyNum')

    if text == '':
        print('\n很抱歉！無法識別\n')
    else:
        print(text)
        
os.system('pause')
