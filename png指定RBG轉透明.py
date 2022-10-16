## png 指定RBG轉透明
##介面完成
##測試完成

from os import sep, path, system, mkdir
from PIL import Image 


###main
while True:
    ##讀取目錄路徑
    pictrueInputDir = str(input('輸入圖片，可以直接拖曳至此：'))
    
    if not path.isfile(pictrueInputDir):
        print('路徑驗證異常，請確認路徑\n')
    else:
        break
    
pictrueOutputDir = str(input('輸出位置，可以直接拖曳至此，預設輸出在目標的目錄：'))
  
if len(pictrueOutputDir) == 0:
    pictrueOutputDir = pictrueInputDir.split("\\")[:-1]
    pictrueOutputDir = "\\".join(pictrueOutputDir)

if not path.isdir(pictrueOutputDir):
    mkdir(pictrueOutputDir)
    print('目錄尚未找到，自動建立新目錄 => ', pictrueOutputDir)


##開啟圖片
pic = Image.open(pictrueInputDir).convert("RGBA")
pic_datas = pic.getdata()
pic_count = {}

for i in pic_datas:
    if pic_count.get(i) == None:
        pic_count[i] = 1
    else:
        pic_count[i] += 1

for i in pic_count.keys():
    pic_count[i] = (pic_count[i], \
                    str(int(pic_count[i] / len(pic_datas) * 100)) + ' %')

print('size ', pic.size)
#print('size ', len(pic_datas))
print('RBGA ', len(pic_count), ' class:\n')

answer = sorted(pic_count.items(), key = lambda x: x[1]) \
         [int(-3 / 100 * len(pic_count)):] 

for i in range(len(answer)):
    if answer[i][0][0] != 0 and \
       answer[i][0][1] != 0 and \
       answer[i][0][2] != 0 and \
       answer[i][0][0] != 1 and \
       answer[i][0][1] != 1 and \
       answer[i][0][2] != 1:
        print(i, ' ', answer[i])

RGB = input('\n指定RBG(以空白間隔)，預設為白色[255, 255, 255]?')

if len(RGB) == 0:
    RGB = [255, 255, 255]
else:
    RGB = RGB.split(', ')

print('\n指定 RGB', RGB)
print('\n開始轉換...\n')

new_pic = []

for i in pic_datas:
    if i[0] == RGB[0] and i[1] == RGB[1] and i[2] == RGB[2]:
        new_pic.append((255, 255, 255, 0))
    else:
        new_pic.append(i)

pic.putdata(new_pic)
pic.save(pictrueOutputDir + sep + 'transparent' + '-'
         + pictrueInputDir.split(sep)[-1].split('.')[0] + '.png', 'PNG')

print('作業完成\n')
system('pause')


