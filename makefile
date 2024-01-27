all:

site: clean
	hugo

dist: site
	scp -r public/* ak2313@webserver.srcf.net:~/public_html

distdraft: clean
	hugo -D --baseURL 'https://ak2313.user.srcf.net/draft/'
	scp -r public/* ak2313@webserver.srcf.net:~/public_html/draft

serve:
	hugo server -D

clean:
	rm -rf public