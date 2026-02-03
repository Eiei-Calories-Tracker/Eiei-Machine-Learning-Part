# **Data Set Management**

# When add new data set

```
dvc add data/v2
git commit -m "add v2"
dvc push
```

# When pull and use

```
git clone ...
```

# **add env file to secrets folder**

- Install (Ensure it's installed)

- **Only First Time**

```
python -m venv .venv
```

- Activate and install

```
.venv\Scripts\activate
pip install "dvc[s3]"
pip install dotenv-cli
dotenv -e ./secrets/.env run dvc pull
```

# How to initalize in case add new dvc

- **Only First Time**

```
dvc init
git commit -m "init dvc"
```

- Add what you want ex. data/v1

```
dvc add data/v1
```

- then commit

```
git add .
git commit -m "track datasets"
```

- Google Drive folder **Only First Time** copy the url (stil bug) https://drive.google.com/drive/folders/XXXXXXXX

```
dvc remote add -d gdrive gdrive://FOLDER_ID
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path secrets/XXX.json
```

- S3 bucket **Only First Time** copy the url

```
dvc remote add -d s3remote s3://bucketname/dvcstore
dvc remote list
```

```
dvc push
```

# Push to GIthub

```
git push origin
```

# Other

```
Get-Content ./secrets/.env | ForEach-Object {
if ($_ -match "=") {
    $name, $value = $_ -split "=",2
    Set-Item -Path env:$name -Value $value
}
}
```

# Directory Prefer now (Th food 50 is now avaliable)

Th food 50

```
data/
 ├── test_fixed/
 │    └── BitterMelon/
 │
 ├── v1/
 │    ├── train/
 │    			└── BitterMelon/
 │    └── val/
 │
 └── v2/
      ├── train/
      └── val/
```

Th food 100

```
data/
 ├── test_fixed/ (สังดึงมาเก็บก่อนเลย)
 │    ├── 0/
 │    ├── 1/..
 │
 ├── v1/
 │    ├── 0/
 │    ├── 1/..
 │
 │
 │
 └── v2/
 │    ├── 0/
 │    ├── 1/..
```
