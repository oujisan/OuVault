# [note] Oguri Cap
---
![oguri](https://a.storyblok.com/f/178900/1920x1080/a2ed2cb993/umamusumecinderellagray_hero.png)

Hello world for oguri cap in `Trecen Academy`
```typescript
const markdowns = await Promise.all(
  files.map(async (file) => {
    const slug = file.name.replace(/\.md$/, "")
      try {
          const res = await api.get(`repos/oujisan/OuVault/contents/${file.name}`)
          const base64 = res.data.content
          const content = atob(base64)
          const { title, category } = extractTitle(content)
  
          return { title, slug, category }
      } catch (err: any) {
          return { title: file.name.replace(/\.md$/, ""), slug, category: 'note' }
      }
  }
)
```
