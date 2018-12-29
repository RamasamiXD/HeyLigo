  function getElementLeft(elm) 
{
    var x = 0;

    //set x to elm’s offsetLeft
    x = elm.offsetLeft;

    //set elm to its offsetParent
    elm = elm.offsetParent;

    //use while loop to check if elm is null
    // if not then add current elm’s offsetLeft to x
    //offsetTop to y and set elm to its offsetParent

    while(elm != null)
    {
        x = parseInt(x) + parseInt(elm.offsetLeft);
        elm = elm.offsetParent;
    }
    return x;
}

function getElementTop(elm) 
{
    var y = 0;

    //set x to elm’s offsetLeft
    y = elm.offsetTop;

    //set elm to its offsetParent
    elm = elm.offsetParent;

    //use while loop to check if elm is null
    // if not then add current elm’s offsetLeft to x
    //offsetTop to y and set elm to its offsetParent

    while(elm != null)
    {
        y = parseInt(y) + parseInt(elm.offsetTop);
        elm = elm.offsetParent;
    }

    return y;
}

function Large(obj,rep)
{
    var imgbox=document.getElementById("imgbox");
    //console.log(obj);
    //console.log(rep);
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
    } else {
        img.attachEvent('onmouseout',Out);
    }             
    imgbox.innerHTML='';
    imgbox.appendChild(title);
    imgbox.appendChild(img);
    //imgbox.style.left=(getElementLeft(obj)-50) +'px';
    //imgbox.style.top=(getElementTop(obj)-50) + 'px';
}

function Out()
{
    document.getElementById("imgbox").style.visibility='hidden';
}



  function make_report() {


    console.log($('#srch-term').val());
    console.log("Entered sort_by_relevance");

    $.ajax({
      data : {
        term:$('#srch-term').val(),
      },
      type : 'POST',
      url : '/report'
    })
    .done(function(data) {  
      if(data.results)
      {          
        console.log(data.results);
        console.log(data.hits);
        $('#noresult').css('display','none');
        $("#merged_plot1").attr('src','../static/img/reports/timeline'+data.hits+'.png');
        $("#RelatedPost1").attr('src','../static/img/reports/pie_'+data.hits+'l.png');
        $("#RelatedPost2").attr('src','../static/img/reports/timeline_'+data.hits+'l.png');
        $("#Author1_img1").attr('src','../static/img/reports/pie_'+data.hits+'h.png');
        $("#Author1_img2").attr('src','../static/img/reports/timeline_'+data.hits+'h.png');
        $("#Author2_img1").attr('src','../static/img/reports/pie_'+data.hits+'v.png');
        $("#Author2_img2").attr('src','../static/img/reports/timeline_'+data.hits+'v.png');
          $(".report").css('display', 'block');
}
    else{
      $('.report').css('display','none');
      $('#noresult').css('display','block');

    }

    });

    event.preventDefault();


  }



  function startDictation() {

    if (window.hasOwnProperty('webkitSpeechRecognition')) {
      document.getElementById('microphone').style.color='red';
      document.getElementById('srch-term').placeholder="Say: Show me more about calibration lines";
      //  document.getElementById('srch-term').style.background-color='green';
      var recognition = new webkitSpeechRecognition();


      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.lang = "en-US";
      recognition.start();

      recognition.onresult = function(e) {
        console.log(e.results[0][0].transcript)
        $('#srch-term').val(e.results[0][0].transcript);
        recognition.stop();
        $('#srch').click(); 
        document.getElementById('microphone').style.color='grey';
      };

      recognition.onerror = function(e) {
        console.log(e)
        recognition.stop();
      }

    }
  }

$(document).ready(function() {
  $('#voice_search').on('submit', function(event) {
      event.preventDefault();
      make_report()

  });



  //-----------------------------------------------------------------------------------------------------------


});