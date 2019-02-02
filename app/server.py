from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://www.dropbox.com/s/w40aftwj7fzp8gg/stage-4_4.pth?dl=1'


model_file_name = 'model'

classes = ['Anas querquedula',
 'Falco subbuteo',
 'Vanellus cinereus',
 'Terpsiphone atrocaudata',
 'Stercorarius parasiticus',
 'Falco amurensis',
 'Tringa erythropus',
 'Aquila heliaca',
 'Himantopus himantopus',
 'Numenius minutus',
 'Lophotriorchis kienerii',
 'Ficedula narcissina',
 'Dicrurus leucophaeus',
 'Arachnothera chrysogenys',
 'Coracina fimbriata\xa0',
 'Numenius madagascariensis',
 'Charadrius veredus',
 'Cuculus saturatus',
 'Puffinus tenuirostris',
 'Tringa guttifer',
 'Pycnonotus melanoleucos',
 'Tringa brevipes',
 'Hemipus hirundinaceus',
 'Lonchura ferruginosa',
 'Emberiza aureola',
 'Anas penelope',
 'Alcedo meninting',
 'Strix leptogrammica',
 'Acridotheres melanopterus',
 'Garrulax canorus',
 'Lonchura striata',
 'Falco severus',
 'Plegadis falcinellus',
 'Accipiter nisus',
 'Ixos malaccensis',
 'Cyornis glaucicomans',
 'Dicaeum chrysorrheum',
 'Calidris acuminata',
 'Motacilla citreola',
 '\xa0Esacus magnirostris',
 'Phylloscopus borealoides',
 'Nisaetus alboniger',
 'Meiglyptes tristis',
 'Asio flammeus',
 'Otus sunia',
 'Phylloscopus fuscatus',
 'Fregata andrewsi',
 'Macheiramphus alcinus',
 'Porzana paykullii',
 'Ninox japonica',
 'Egretta eulophotes',
 'Rhaphidura leucopygialis',
 'Numenius arquata',
 'Anastomus oscitans',
 'Agropsar philippensis',
 'Circus cyaneus',
 'Trichastoma rostratum',
 'Calidris alpina',
 'Treron olax',
 'Cyanoptila cumatilis',
 'Phylloscopus inornatus',
 'Anthus richardi',
 'Anas clypeata',
 'Hierococcyx sparverioides',
 'Fregata ariel',
 'Hemiprocne comata',
 'Arachnothera crassirostris',
 'Cymbirhynchus macrorhynchos',
 'Lyncornis temminckii',
 'Rostratula benghalensis',
 'Pastor roseus',
 'Muscicapa williamsoni',
 'Microhierax fringillarius',
 'Chroicocephalus brunnicephalus',
 'Monticola gularis',
 'Anhinga melanogaster',
 'Iole olivacea',
 'Gallinago megala',
 'Circaetus gallicus',
 'Calidris tenuirostris',
 'Gyps himalayensis',
 'Circus aeruginosus',
 'Saxicola stejnegeri',
 'Dicaeum agile',
 'Clamator jacobinus',
 'Calidris melanotos',
 'Gelochelidon nilotica',
 'Terpsiphone paradisi',
 'Aythya fuligula',
 'Pitta megarhyncha',
 'Pachycephala cinerea',
 'Hydroprogne caspia',
 'Anas acuta',
 'Sterna dougallii',
 'Muscicapa griseisticta',
 'Fulica atra',
 'Passer domesticus',
 'Eurynorhynchus pygmeus',
 'Hydrophasianus chirurgus',
 'Tringa ochropus',
 'Stercorarius pomarinus',
 'Geokichla sibirica',
 'Glareola lactea',
 'Leptoptilos javanicus',
 'Ardeola grayii',
 'Circus melanoleucos',
 'Calidris canutus',
 'Prionochilus thoracicus',
 'Hirundapus caudacutus',
 'Phalaropus lobatus',
 'Gorsachius melanolophus',
 'Anthreptes simplex',
 'Ficedula elisae',
 'Mulleripicus pulverulentus',
 'Scolopax rusticola',
 'Monticola solitarius',
 'Spodiopsar sericeus ',
 'Anas strepera',
 'Iduna caligata',
 'Anthracoceros malayanus',
 'Delichon dasypus',
 'Hypothymis azurea',
 'Charadrius hiaticula',
 'Pericrocotus speciosus',
 'Tachybaptus ruficollis',
 'Accipiter virgatus',
 'Bubo sumatranus',
 'Dryocopus javensis',
 'Nettapus coromandelianus',
 'Ducula badia',
 'Stercorarius longicaudus',
 'Ploceus manyar',
 'Eumyias thalassinus',
 'Treron fulvicollis',
 'Lonchura oryzivora',
 'Anas crecca',
 'Pycnonotus atriceps',
 'Cyornis rufigastra',
 'Chrysococcyx maculatus',
 'Sula sula',
 'Rallina eurizonoides',
 'Sula leucogaster',
 'Philomachus pugnax',
 'Calidris temminckii',
 'Chrysococcyx basalis',
 'Milvus migrans',
 'Malacopteron magnirostre',
 'Spilornis cheela',
 'Acridotheres cristatellus',
 'Cyanoptila cyanomelana',
 'Larus fuscus',
 'Heliopais personata',
 'Aquila nipalensis',
 'Clanga clanga',
 'Falco naumanni']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)

