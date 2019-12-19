



## Setup

```shell script
docker build -t msarthur/w2v-rest .
```


## Publish image

```shell script
docker push msarthur/w2v-rest:latest
```


## RUN

```shell script
docker run --name w2v-rest -p 5001:5001 msarthur/w2v-rest
```

## Testing
```shell script
curl -XGET 'http://localhost:5001/w2v/similarity?a=Have+you+ever+tried+to+used+the+method+string.split%28%22hello+world%22%29&b=try+using+the+string.split+method' | json_pp
```

Response should look like:

```json
{
   "a" : "Have you ever tried to used the method string.split(\"hello world\")",
   "b" : "try using the string.split method",
   "err" : null,
   "sim" : 0.777944326400757
}
``` 