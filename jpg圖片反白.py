## jpg 圖片反白
##介面完成
##測試完成

from os import sep, path, system, mkdir
from PIL import Image 
import numpy as np


###main
##讀取目錄路徑

pictrueInputDir = str(input('輸入圖片，可以直接拖曳至此：'))
if not path.isfile(pictrueInputDir):
    print('路徑驗證異常，請確認路徑')
    
pictrueOutputDir = str(input('輸出位置，可以直接拖曳至此，預設輸出在目標的目錄：'))
    
if len(pictrueOutputDir) == 0:
    pictrueOutputDir = pictrueInputDir.split("\\")[:-1]
    pictrueOutputDir = "\\".join(pictrueOutputDir)

if not path.isdir(pictrueOutputDir):
    mkdir(pictrueOutputDir)
    print('目錄尚未找到，自動建立新目錄 => ', pictrueOutputDir)

print('\n開始轉換...\n')
##開啟圖片並轉換

pic = Image.open(pictrueInputDir)
pic_arr = np.array(pic)
fpic_arr = 255 - pic_arr

fixPic = Image.fromarray(fpic_arr)
fixPic.save(pictrueOutputDir + sep + 'fix-'
            + pictrueInputDir.split(sep)[-1].split('.')[0] + '.png')

print('作業完成\n')
system('pause')


