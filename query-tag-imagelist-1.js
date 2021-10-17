const AWS = require('aws-sdk');
AWS.config.update({ region: process.env.AWS_REGION })
 const dynamo = new AWS.DynamoDB.DocumentClient();
 const s3 = new AWS.S3();
 exports.handler = async(event, context, callback) => {
  console.log('remaining time =', context.getRemainingTimeInMillis());
  console.log('functionName =', context.functionName);
  console.log('AWSrequestID =', context.awsRequestId);
  
  var body = {};
  //prepare statuscode and headers for response to API
  let statusCode = '200';
  const headers = {'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  "Access-Control-Allow-Headers": "https://amplify.dqxzfllk3x5zy.amplifyapp.com/, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers", 
  "Access-Control-Allow-Credentials": "true",
  "Access-Control-Allow-Methods": "GET,HEAD,OPTIONS,POST,PUT"};
  const tableName = 'tags';
  try {
   switch (event.httpMethod) {
    //Delete case for deleting images from dynamodb and s3
    case 'DELETE':
     var data ={};
     // get the id passed from the user
     data.id = event.body && JSON.parse(event.body.trim()).id;
     console.log(data.id);
     console.log(data.id.split('/').pop());
     var s3_key ='public/' + data.id.split('/').pop();
     if (data.id){
      // locate object in s3 bucket and delete it 
      let params1 = {Bucket: 'tagtagstorage10413-dev', Key: s3_key};
      s3.deleteObject(params1, function(err, data) {
       if (err) {
        //error
        console.log(err, err.stack);  
       }
       else{
        // deleted
        console.log('delete', data);
       }                      
        });
        // Locate object in dynamodb and delete it
      let params = {TableName: tableName, Key: data};
      body = await dynamo.delete(params).promise();
     }
     break;
     
     //post method to add tags to an image
     case 'POST':
      var tags1 = [];
       var tags=[];
      data ={};
      // get the id passed from api gateway
      data.id = event.body && JSON.parse(event.body.trim()).id;
      console.log(data.id);
      //get the tags from api gateway
      var tag_part1 =JSON.parse(event.body.trim()).tags;
      tags1 = tag_part1.split(",");
      if (data.id){
       //Find the item in dynamodb by using key/imageID
       let params = {TableName: tableName, Key: data};
       var result =  await dynamo.get(params).promise();
       if(result.Item.tags){
        const tags = JSON.parse(JSON.stringify(result.Item.tags));
        console.log(tags);
        //Iterate through the tags and check if they exist on the image
        for(i = 0; i < tags1.length; i++){
        if (tags.includes(tags1[i])){
         console.log("Tag Exists");
        }
        else{
         //if tag does not exist it will be added to the tags list
         tags.push(tags1[i]);
        }
        }
        data.tags=dynamo.createSet(tags);
       }
       else{
      
       for(i = 0; i < tags1.length; i++){
       tags.push(tags1[i]);
       }
       data.tags=dynamo.createSet(tags);
       }
       
       //Add the tags to the image in dynamodab
       let params1 = {TableName: tableName, Item: data};
       body = await dynamo.put(params1).promise();

      }
      break;
      
      // get method to get images with matching tags 
    case 'GET':
     var tags = [];
     let body1;
     let body2;
     console.log(event);
     //When get is called
     if (event.queryStringParameters) {
      // get tags from aoi gateway
      var tag_part =(event.queryStringParameters.tags).trim();
      tags = tag_part.split(",");
      console.log(tags);
      var i,j;
      body.links =[];
      //iterate through the tags
      for (i = 0; i < tags.length; i++) {
       console.log(tags[i]);
       //set the oaraneters to search for 
       let params = {TableName: tableName, FilterExpression: "contains (tags, :category1)",ExpressionAttributeValues: { ":category1": tags[i].trim() },};
      //scan dynamodb for ids with matching tags
       body1 = await dynamo.scan(params).promise();
       for(j = 0; j < body1.Items.length; j++) {
        //check if tags are not empty 
        if(body1.Items != []){
         //check if the id is already in the reposnse body
         if(body.links.includes(body1.Items[j].id)){
         console.log("Exists");
        }
        else{
         //if id does not exist in response body add it to the response body 
         body.links.push(body1.Items[j].id)
        
        }
        }
        
        
       }
      }
     console.log(body.links.length)  
     }
     else{
      //commit changes to dynamoDB
      body = await dynamo.scan({ TableName: tableName }).promise();
      
     
     break;
    }
   }
  }
  catch (err) {
   statusCode = '400';
   body = err.message;
  }
  finally {body = JSON.stringify(body);}
  //Return the status code, body and headers to the user
  return {statusCode, body,headers,};
 };
