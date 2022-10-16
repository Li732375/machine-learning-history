##批量圖片辨識
##批量圖片重組>>影像處理>>OCR辨識並重新命名檔名
##介面完成
##測試完成

from os import sep, path, system, mkdir, listdir
from PIL import Image, ImageEnhance, ImageFilter
from time import time, localtime
import pytesseract 


##裁切等分
scale = 7
##指定副檔名
pictureFileExtension = 'jpg'
##亮度增強值
l_fac = 16
##銳度增強值
c_fac = 10
##中值濾波尺寸
size = 3

###main
##讀取目錄路徑
#'F:\\python作品\\機器學習\\test'
pictrueInputDir = str(input('輸入圖片目錄，可以直接拖曳至此：'))

if not path.isdir(pictrueInputDir):
    print('路徑驗證異常，請確認路徑')
    
pictrueOutputDir = str(input('輸出位置，可以直接拖曳至此，預設輸出在目標的目錄：'))
    
if len(pictrueOutputDir) == 0:
    t = str(localtime(time()).tm_hour) + '_' + str(localtime(time()).tm_min) + '_' + str(localtime(time()).tm_sec)
    pictrueOutputDir = pictrueInputDir + '_dataset_' + t
    pictrueOutputDirAdj = pictrueInputDir + '_adjustData_' + t
    
if not path.isdir(pictrueOutputDir):
    mkdir(pictrueOutputDir)
    print('目錄尚未找到，自動建立新目錄 => ', pictrueOutputDir)

if not path.isdir(pictrueOutputDirAdj):
    mkdir(pictrueOutputDirAdj)
    print('目錄尚未找到，自動建立新目錄 => ', pictrueOutputDirAdj)

# 逐一查詢檔案清單
for item in listdir(pictrueInputDir):
    #緩衝一下
    sleep(1.5)
    
    target = path.join(pictrueInputDir, item)
    
    if path.isdir(target):
    #使用 isdir 檢查是否為目錄
    #使用 join 的方式把路徑與檔案名稱串起來
        print('\n目錄: ', item)    
    
    elif path.isfile(target):
    #使用isfile判斷是否為檔案
        print('\n- 發現檔案: ', item)

        if path.splitext(item)[-1] == '.' + pictureFileExtension:
            pic = Image.open(target)

            width, height = pic.size
            
            #求出分割比例單位
            part = int(height / scale)

            ##分割成上下半
            #crop((left, upper, right, lower))
            halfUp = pic.crop((0, 0, width, part))
            halfDown = pic.crop((0, part, width, height))
            print('- 分割完成')

            #重組並建立空白圖片
            newPic = Image.new('RGB', (width, height))

            newPic.paste(halfDown, (0, 0, width, height - part))
            newPic.paste(halfUp, (0, height - part , width, height))
            print('- 重組完成')

            ##影像處理
            #調亮
            newPic = ImageEnhance.Brightness(newPic).enhance(l_fac)

            #提升銳利度
            newPic = ImageEnhance.Contrast(newPic).enhance(c_fac)

            #中值濾波
            newPic = newPic.filter(ImageFilter.MedianFilter(size))

            print('- 影像處理完成')

            ##OCR辨識並修改檔名
            answer = pytesseract.image_to_string(newPic,
                                                 config = '--psm 7 digits_onlyNum')

            if len(answer) >= 4:
                answer = answer[:4]

            print('- 辨識結果：', answer)
            
            try:
                pic.save(pictrueOutputDir + sep + answer + '.jpg',
                     quality = 100)

                newPic.save(pictrueOutputDirAdj + sep + answer + '_.jpg',
                     quality = 100)
                
            except OSError:
                pic.save(pictrueOutputDir + sep + item + '.jpg',
                     quality = 100)
                
                newPic.save(pictrueOutputDirAdj + sep + item + '_.jpg',
                     quality = 100)
                
            except:
                traceback.print_exc()

print('\n作業完成\n')
system('pause')



