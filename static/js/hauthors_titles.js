function getElementLeft(elm){
    var x = 0;
    x = elm.offsetLeft;
    elm = elm.offsetParent;
    while(elm != null){
        x = parseInt(x) + parseInt(elm.offsetLeft);
        elm = elm.offsetParent;
    }
    return x;
}

function getElementTop(elm){
    var y = 0;
    y = elm.offsetTop;
    elm = elm.offsetParent;
    while(elm != null){
        y = parseInt(y) + parseInt(elm.offsetTop);
        elm = elm.offsetParent;
    }
    return y;
}

function Large(obj,rep){
    var imgbox=document.getElementById("imgbox");
    imgbox.style.visibility='visible';
    var img = document.createElement("img");
    var title = document.createElement("H6");                       // Create a <p> node
    var t = document.createTextNode(rep); 
    title.appendChild(t); 
    img.src=obj.src;
    img.style.width='300px';
    img.style.height='300px';
    if(img.addEventListener){
        img.addEventListener('mouseout',Out,false);
    }
    else{
        img.attachEvent('onmouseout',Out);
    }             
    imgbox.innerHTML='';
    imgbox.appendChild(title);
    imgbox.appendChild(img);
}

function Out(){
    document.getElementById("imgbox").style.visibility='hidden';
}

function sort_by_time(){
    document.getElementById('rel').className='btn btn-alert';
    document.getElementById('rec').className='btn btn-success';
    document.getElementById('authrel').className='btn btn-alert';
    document.getElementById('authrec').className='btn btn-success';
    
    
    $.ajax({
        data : {
            term:$('#srch-term').val(),
        },
        type : 'POST',
        url : '/hsearch_authors_titles'
    }).done(function(data) { 
        $('#auth1').empty();
        $('#auth2').empty();
        $('#auth3').empty();
        $('#auth4').empty();
        $('#authorlist').empty();
        $('#dccauth1').empty();
        $('#dccauth2').empty();
        $('#dccauth3').empty();
        $('#dccauth4').empty();
        $('#dccauthorlist').empty(); 
        if(data.results){
            var top_searches = data.top_searches_sorted;
            var author_titles = data.author_titles_sorted;
            var names =  data.names;
            var images = data.images_sorted;
            if (images.length>0){
                $('#img').empty();
                for (index = 0; index < images.length; ++index){
                    var title = images[index][2].replace("'",'')
                    title = title.replace("\"","")
                    title = title.replace(/\\/g,"")
                    var func_str = "Large(this,\""+title+"\")"
                    $('#img').append("<a target='_blank' href='https://alog.ligo-la.caltech.edu/aLOG/iframeSrc.php?authExpired=&content=1&step=&callRep="+images[index][1]+"&startPage=&preview=&printCall=&callUser=&addCommentTo=&callHelp=&callFileType=#'><img src='"+images[index][0]+"' onmouseout='Out()' onmouseover='"+func_str+"' height='60px' width='60px'></a>")
                    $("#img").css('display', 'block');
                }
            }
            else{
                $("#img").css('display', 'none');
            }
            $('#RelatedPost').empty();
            for (index = 0; index < top_searches.length; ++index){
                $('#RelatedPost').append("<a target='_blank'  style='background-color: "+top_searches[index][4]+";' class='list-group-item' href='https://alog.ligo-wa.caltech.edu/aLOG/iframeSrc.php?authExpired=&content=1&step=&callRep="+top_searches[index][0]+"&startPage=&preview=&printCall=&callUser=&addCommentTo=&callHelp=&callFileType=#'><h4 class='list-group-item-heading title_tag'>"+top_searches[index][1]+"</h4><p class='list-group-item-text'></p>"+top_searches[index][2]+"<p class='list-group-item-text'></p>"+top_searches[index][5]+"<p class='list-group-item-text date_tag'>"+top_searches[index][3]+"</p></a>")
            }
            var i = 0;
            for (var key in author_titles){
                if(author_titles[key]==null)
                    continue;
                i = i + 1;
                var t = "";
                if(i==3)
                    $('#authorlist').append('<br>');
                $('#authorlist').append('<button onclick="changeauth('+i+');" class="btn btn-alert" style="max-width:200px;overflow:hidden;" id="authbutton'+i+'">'+names[i-1]+'</button>');
                for (index = 0; index < author_titles[key].length; ++index){
                    t+="<a target='_blank'  style='background-color: "+author_titles[key][index][4]+";' class='list-group-item' href='https://alog.ligo-wa.caltech.edu/aLOG/iframeSrc.php?authExpired=&content=1&step=&callRep="+author_titles[key][index][0]+"&startPage=&preview=&printCall=&callUser=&addCommentTo=&callHelp=&callFileType=#'><h4 class='list-group-item-heading title_tag'>"+author_titles[key][index][1]+"</h4><p class='list-group-item-text date_tag'>"+author_titles[key][index][3]+"</p></a>";
                }
                document.getElementById('auth'+i).value = t;
            }
            document.getElementById('Auth').innerHTML=document.getElementById('auth1').value;
            document.getElementById('authbutton1').className='btn btn-success';
        }
        else{
            $("#img").css('display', 'none');
            $('#RelatedPost').empty();
            $('#RelatedPost').append("<div class='alert alert-danger'><p>Sorry No relevant results found</p></div>");
            $('#authorlist').empty();
            $('#Auth').empty();
            $('#Auth').append("<div class='alert alert-danger'><p>Sorry No relevant results found</p></div>")
        }
        if(data.dcc_result.length != 0){
            var dcc_result = data.dcc_result;

            $('#DCC').empty();
            for (index = 0; index < dcc_result.length; ++index) {
                $('#DCC').append("<a  target='_blank' style='background-color: "+dcc_result[index][4]+";' class='list-group-item' href='https://dcc.ligo.org"+dcc_result[index][2]+"'><h4 class='list-group-item-heading title_tag'>"+dcc_result[index][0]+"</h4><p class='list-group-item-text'></p>"+dcc_result[index][1]+"</a>")
            }
            var dccnames = data.dcc_author_name;
            var dcc_author_titles = data.dcc_author;
            var i = 0;
            for (var key in dcc_author_titles){
                if(dcc_author_titles[key]==null)
                    continue;
                i = i + 1;
                var t = "";
                if(i==3)
                    $('#dccauthorlist').append('<br>');
                $('#dccauthorlist').append('<button onclick="dccchangeauth('+i+');" class="btn btn-alert" style="max-width:200px;overflow:hidden;" id="dccauthbutton'+i+'">'+dccnames[i-1]+'</button>');
                for (index = 0; index < dcc_author_titles[key].length; ++index){
                    t+="<a  target='_blank' style='background-color: "+dcc_author_titles[key][index][4]+";' class='list-group-item' href='https://dcc.ligo.org"+dcc_author_titles[key][index][2]+"'><h4 class='list-group-item-heading title_tag'>"+dcc_author_titles[key][index][0]+"</h4><p class='list-group-item-text'></p>"+dcc_author_titles[key][index][1]+"</a>";
                }
                document.getElementById('dccauth'+i).value = t;
            }
            document.getElementById('dccAuth').innerHTML=document.getElementById('dccauth1').value;
            document.getElementById('dccauthbutton1').className='btn btn-success';
        }
        else{
            $('#DCC').empty();
            $('#DCC').append("<div class='alert alert-danger'><p>Sorry No relevant results found</p></div>");
            $('#dccauthorlist').empty();
            $('#dccAuth').empty();
            $('#dccAuth').append("<div class='alert alert-danger'><p>Sorry No relevant results found</p></div>")
        }
    });
    event.preventDefault();
}

function changeauth(i){
    document.getElementById('authbutton1').className='btn btn-alert';
    document.getElementById('authbutton2').className='btn btn-alert';
    document.getElementById('authbutton3').className='btn btn-alert';
    document.getElementById('authbutton4').className='btn btn-alert';
    document.getElementById('authbutton'+i).className='btn btn-success';
    document.getElementById('Auth').innerHTML=document.getElementById('auth'+i).value;
}

function dccchangeauth(i){
    document.getElementById('dccauthbutton1').className='btn btn-alert';
    document.getElementById('dccauthbutton2').className='btn btn-alert';
    document.getElementById('dccauthbutton3').className='btn btn-alert';
    document.getElementById('dccauthbutton4').className='btn btn-alert';
    document.getElementById('dccauthbutton'+i).className='btn btn-success';
    document.getElementById('dccAuth').innerHTML=document.getElementById('dccauth'+i).value;
}

function sort_by_relevance() {
    document.getElementById('rel').className='btn btn-success';
    document.getElementById('rec').className='btn btn-alert';
    document.getElementById('authrel').className='btn btn-success';
    document.getElementById('authrec').className='btn btn-alert';
    $.ajax({
        data : {
            term:$('#srch-term').val(),
        },
        type : 'POST',
        url : '/hsearch_authors_titles'
    }).done(function(data){  
        $('#auth1').empty();
        $('#auth2').empty();
        $('#auth3').empty();
        $('#auth4').empty();
        $('#authorlist').empty();
        $('#dccauth1').empty();
        $('#dccauth2').empty();
        $('#dccauth3').empty();
        $('#dccauth4').empty();
        $('#dccauthorlist').empty();
        if(data.results != null && data.results.length != 0){
            var top_searches = data.top_searches;
            var author_titles = data.author_titles;
            var names =  data.names;
            var images = data.images;
            if(images.length>0){
                
                $('#img').empty();
                for (index = 0; index < images.length; ++index){
                    var title = images[index][2].replace("'",'')
                    title = title.replace("\"","")
                    title = title.replace(/\\/g,"")
                    var func_str = "Large(this,\""+title+"\")"
                    $('#img').append("<a target='_blank' href='https://alog.ligo-la.caltech.edu/aLOG/iframeSrc.php?authExpired=&content=1&step=&callRep="+images[index][1]+"&startPage=&preview=&printCall=&callUser=&addCommentTo=&callHelp=&callFileType=#'><img src='"+images[index][0]+"' onmouseout='Out()' onmouseover='"+func_str+"' height='60px' width='60px'></a>")
                    $("#img").css('display', 'block');
                }
            }
            else{
                $("#img").css('display', 'none');
            }
            $('#RelatedPost').empty();
            for (index = 0; index < top_searches.length; ++index) {
                $('#RelatedPost').append("<a target='_blank'  style='background-color: "+top_searches[index][4]+";' class='list-group-item' href='https://alog.ligo-wa.caltech.edu/aLOG/iframeSrc.php?authExpired=&content=1&step=&callRep="+top_searches[index][0]+"&startPage=&preview=&printCall=&callUser=&addCommentTo=&callHelp=&callFileType=#'><h4 class='list-group-item-heading title_tag'>"+top_searches[index][1]+"</h4><p class='list-group-item-text'></p>"+top_searches[index][2]+"<p class='list-group-item-text'></p>"+top_searches[index][5]+"<p class='list-group-item-text date_tag'>"+top_searches[index][3]+"</p></a>")
            }
            var i = 0;
            for (var key in author_titles){
                if(author_titles[key]==null)
                    continue;
                i = i + 1;
                var t = "";
                if(i==3)
                    $('#authorlist').append('<br>');
                $('#authorlist').append('<button onclick="changeauth('+i+');" style="max-width:200px;overflow:hidden;" class="btn btn-alert" id="authbutton'+i+'">'+names[i-1]+'</button>');
                for (index = 0; index < author_titles[key].length; ++index){
                    t+="<a target='_blank'  style='background-color: "+author_titles[key][index][4]+";' class='list-group-item' href='https://alog.ligo-wa.caltech.edu/aLOG/iframeSrc.php?authExpired=&content=1&step=&callRep="+author_titles[key][index][0]+"&startPage=&preview=&printCall=&callUser=&addCommentTo=&callHelp=&callFileType=#'><h4 class='list-group-item-heading title_tag'>"+author_titles[key][index][1]+"</h4><p class='list-group-item-text date_tag'>"+author_titles[key][index][3]+"</p></a>";
                }
                document.getElementById('auth'+i).value = t;
            }
            document.getElementById('Auth').innerHTML=document.getElementById('auth1').value;
            document.getElementById('authbutton1').className='btn btn-success';
        }
        else{
            $("#img").css('display', 'none');
            $('#RelatedPost').empty();
            $('#RelatedPost').append("<div class='alert alert-danger'><p>Sorry No relevant results found</p></div>");
            $('#authorlist').empty();
            $('#Auth').empty();
            $('#Auth').append("<div class='alert alert-danger'><p>Sorry No relevant results found</p></div>")
        }
        if(data.dcc_result != null && data.dcc_result.length != 0){
            var dcc_result = data.dcc_result;

            $('#DCC').empty();
            for (index = 0; index < dcc_result.length; ++index) {
                $('#DCC').append("<a  target='_blank' style='background-color: "+dcc_result[index][4]+";' class='list-group-item' href='https://dcc.ligo.org"+dcc_result[index][2]+"'><h4 class='list-group-item-heading title_tag'>"+dcc_result[index][0]+"</h4><p class='list-group-item-text'></p>"+dcc_result[index][1]+"</a>")
            }
            var dccnames = data.dcc_author_name;
            var dcc_author_titles = data.dcc_author;
            var i = 0;
            for (var key in dcc_author_titles){
                if(dcc_author_titles[key]==null)
                    continue;
                i = i + 1;
                var t = "";
                if(i==3)
                    $('#dccauthorlist').append('<br>');
                $('#dccauthorlist').append('<button onclick="dccchangeauth('+i+');" class="btn btn-alert" style="max-width:200px;overflow:hidden;" id="dccauthbutton'+i+'">'+dccnames[i-1]+'</button>');
                for (index = 0; index < dcc_author_titles[key].length; ++index){
                    t+="<a  target='_blank' style='background-color: "+dcc_author_titles[key][index][4]+";' class='list-group-item' href='https://dcc.ligo.org"+dcc_author_titles[key][index][2]+"'><h4 class='list-group-item-heading title_tag'>"+dcc_author_titles[key][index][0]+"</h4><p class='list-group-item-text'></p>"+dcc_author_titles[key][index][1]+"</a>";
                }
                document.getElementById('dccauth'+i).value = t;
            }
            document.getElementById('dccAuth').innerHTML=document.getElementById('dccauth1').value;
            document.getElementById('dccauthbutton1').className='btn btn-success';
        }
        else{
            $('#DCC').empty();
            $('#DCC').append("<div class='alert alert-danger'><p>Sorry No relevant results found</p></div>");
            $('#dccauthorlist').empty();
            $('#dccAuth').empty();
            $('#dccAuth').append("<div class='alert alert-danger'><p>Sorry No relevant results found</p></div>")
        }
    });
    event.preventDefault();
}

function startDictation() {
    if (window.hasOwnProperty('webkitSpeechRecognition')){
        document.getElementById('microphone').style.color='red';
        document.getElementById('srch-term').placeholder="Say: Show me more about calibration lines";
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-US";
        recognition.start();
        recognition.onresult = function(e) {
            
            $('#srch-term').val(e.results[0][0].transcript);
            recognition.stop();
            $('#srch').click(); 
            document.getElementById('microphone').style.color='grey';
        };
        recognition.onerror = function(e) {
            
            recognition.stop();
        }
    }
}

$(document).ready(function(){
    $('#voice_search').on('submit', function(event) {
        event.preventDefault();
        sort_by_relevance()
    });
});
