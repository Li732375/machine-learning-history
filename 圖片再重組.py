##圖片重組
##介面完成
##測試完成

from os import sep, path, system, mkdir
from PIL import Image 


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

##裁切等分
scale = input('分割等分數? (預設為 7)')

if len(scale) == 0:
    scale = 7
else:
    scale = int(scale)

print('\n開始轉換...\n')
##開啟圖片
pic = Image.open(pictrueInputDir)

print(pic.size)
width, height = pic.size

#求出分割比例單位
part = int(height / scale)

##分割成上下半
#crop((left, upper, right, lower))
halfUp = pic.crop((0, 0, width, part))
halfDown = pic.crop((0, part, width, height))
print('分割完成\n')

##重組
#建立空白圖片
newPic = Image.new('RGB', (width, height))

newPic.paste(halfDown, (0, 0, width, height - part))
newPic.paste(halfUp, (0, height - part , width, height))
print('重組完成\n')

newPic.save(pictrueOutputDir + sep + 'reb' + str(scale) + '-'
            + pictrueInputDir.split(sep)[-1].split('.')[0] + '.jpg',
            quality = 100)

print('作業完成\n')
system('pause')


