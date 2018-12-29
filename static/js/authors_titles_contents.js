$(document).ready(function() {

	$('form').on('submit', function(event) {
		console.log($('#srch-term').val());
		$.ajax({
			data : {
				term:$('#srch-term').val(),
			},
			type : 'POST',
			url : '/search_authors_titles_contents'
		})
		.done(function(data) {	
			if(data.results)
			{
				var top_searches = data.top_searches;
				var author_titles = data.author_titles;
				var names =  data.names;
				$('#RelatedPost').empty();
				for (index = 0; index < top_searches.length; ++index) {
					console.log(top_searches[index])
					$('#RelatedPost').append("<li><p><a href='https://alog.ligo-la.caltech.edu/aLOG/iframeSrc.php?authExpired=&content=1&step=&callRep="+top_searches[index][0]+"&startPage=&preview=&printCall=&callUser=&addCommentTo=&callHelp=&callFileType=#'>"+top_searches[index][1]+"---<b>"+top_searches[index][2]+"</b></a></p></li>")
    //console.log(top_searches[index]);
}
console.log(names)
$('#Author1').empty();
$('#Author2').empty();
$('#Author3').empty();
var i = 0;
for (var key in author_titles){
	i = i + 1;
	$('#Author'+i+'_name').html(names[i-1]);
	for (index = 0; index < author_titles[key].length; ++index) {
					$('#Author'+i).append("<li><p><a href='https://alog.ligo-la.caltech.edu/aLOG/iframeSrc.php?authExpired=&content=1&step=&callRep="+author_titles[key][index][0]+"&startPage=&preview=&printCall=&callUser=&addCommentTo=&callHelp=&callFileType=#'>"+author_titles[key][index][1]+"</a></p></li>")
    //console.log(author_titles[key][index]);
}
}
}
		else{
			$('#RelatedPost').empty();
			$('#Author1_name').html('No Authors found');
			$('#Author2_name').html('No Authors found');
			$('#Author3_name').html('No Authors found');
			$('#RelatedPost').append("<li><p><a href='#'>Oops:Stop Kidding me :P</a></p></li>")
			$('#Author1').empty();
			$('#Author1').append("<li><p><a href='#'>Oops:Stop Kidding me :P</a></p></li>");
			$('#Author2').empty();
			$('#Author2').append("<li><p><a href='#'>Oops:Stop Kidding me :P</a></p></li>");
			$('#Author3').empty();
			$('#Author3').append("<li><p><a href='#'>Oops:Stop Kidding me :P</a></p></li>");

		}

		});

		event.preventDefault();

	});

});