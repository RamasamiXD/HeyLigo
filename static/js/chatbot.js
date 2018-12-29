var opened =0;
function openchatbot(argument) {
    document.getElementById('chatbutton').style.display='none';
    document.getElementById('chatbox').style.display='block';
    opened = 1;

}
function closechatbot(argument) {
    document.getElementById('chatbutton').style.display='block';
    document.getElementById('chatbox').style.display='none';
}

var active = 0;

setInterval(function(){
    if (opened == 0){
        if (active == 0){ 
            active = 1; 
            document.getElementById('chatbutton').style.backgroundColor = "red";
        }
        else{
            active = 0; 
            document.getElementById('chatbutton').style.backgroundColor = "blue"; 
        }
    }
    else{
        document.getElementById('chatbutton').style.backgroundColor = "red";
        return;
    }
}, 1000);




function startbot(){
    chatask("hello");
}
var queryterm = "";

function Savefeedback(query){
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            console.log("saved feedback");
        }
    };
    xhttp.open("POST", "../savefeedback", true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send("query="+query);
    return false;
}
function response(text,query){
    var objDiv = document.getElementById ("messagearea");
    switch(text.split(' ')[0]){
        case '@yesno':{
            objDiv.innerHTML += "<div class='row'><div class='reply'>"+text.substr(text.indexOf(" ") + 1);+"</div></div>";
            objDiv.innerHTML += "<div class='row'><div class='querybutton col-xs-6' onclick='chatask(\"yes\")'>Yes</div><div class='querybutton col-xs-6' onclick='chatask(\"no\")'>No</div></div>";
            objDiv.scrollTop = objDiv.scrollHeight;
        }break;
        case "@Fallback": {
            queryterm = query;
            objDiv.innerHTML += "<div class='row'><div class='reply'>"+text.substr(text.indexOf(" ") + 1);+"</div></div>";
            objDiv.innerHTML += "<div class='row'><div class='querybutton col-xs-6' onclick='chatask(\"yes\")'>Yes</div><div class='querybutton col-xs-6' onclick='chatask(\"no\")'>No</div></div>";
            objDiv.scrollTop = objDiv.scrollHeight;
        }break;
        case "@Search": {
            objDiv.innerHTML += "<div class='row'><div class='reply'>"+text.substr(text.indexOf(" ") + 1);+"</div></div>";
            objDiv.scrollTop = objDiv.scrollHeight;
            document.getElementById('srch-term').value = queryterm;
            sort_by_relevance();
        }break;
        case "@SaveFeedback": {
            Savefeedback(query);
            objDiv.innerHTML += "<div class='row'><div class='reply'>"+text.substr(text.indexOf(" ") + 1);+"</div></div>";
            objDiv.scrollTop = objDiv.scrollHeight;
        }break;
        default:{
            objDiv.innerHTML += "<div class='row'><div class='reply'>"+text+"</div></div>";
            objDiv.scrollTop = objDiv.scrollHeight;
        }
    }
}
function chatask(text){
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            response(this.responseText,text);
        }
    };
    xhttp.open("POST", "../sendchat", true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send("query="+text+"&sessionId="+document.getElementById('sessionId').value);
    return false;
}
function chat(){
    if(document.getElementById("inputarea").value == "" || document.getElementById("inputarea").value == null){
        return false;
    }
    var xhttp = new XMLHttpRequest();
    var query = document.getElementById("inputarea").value;
    document.getElementById("inputarea").value = "";
    var objDiv = document.getElementById ("messagearea");
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            response(this.responseText,query);
        }
    };
    objDiv.innerHTML += "<div class='row'><div class='query'>"+query+"</div></div>";
    xhttp.open("POST", "../sendchat", true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send("query="+query+"&sessionId="+document.getElementById('sessionId').value);
    return false;
    }
